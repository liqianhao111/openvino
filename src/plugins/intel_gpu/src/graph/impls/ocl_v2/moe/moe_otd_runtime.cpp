// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace ov::intel_gpu::ocl::moe_otd {

void OtdPerfCounters::dump() const {
    const auto executions = moe_executions.load(std::memory_order_relaxed);
    const auto hits = gpu_hits.load(std::memory_order_relaxed);
    const auto misses = gpu_misses.load(std::memory_order_relaxed);
    const auto total = hits + misses;
    const auto loads = tensor_load_count.load(std::memory_order_relaxed);
    std::cerr << "[OTD_PERF] moe_executions=" << executions << ", gpu_hits=" << hits << ", gpu_misses=" << misses
              << ", gpu_hit_rate=" << (total > 0 ? 100.0 * hits / total : 0.0) << "%"
              << ", tensor_loads=" << loads << ", avg_disk_io_us=" << (loads > 0 ? disk_io_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
              << ", avg_transpose_us=" << (loads > 0 ? transpose_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
              << ", avg_gpu_copy_us=" << (loads > 0 ? gpu_copy_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
              << ", finish_wait_ms=" << finish_wait_ns.load(std::memory_order_relaxed) / 1000000
              << ", topk_readback_ms=" << topk_readback_ns.load(std::memory_order_relaxed) / 1000000
              << ", index_write_ms=" << index_write_ns.load(std::memory_order_relaxed) / 1000000
              << ", batched_gemv_prepare_ms=" << batched_gemv_prepare_ns.load(std::memory_order_relaxed) / 1000000
              << ", prefill_fallbacks=" << prefill_fallbacks.load(std::memory_order_relaxed)
              << ", prefill_fallback_ms=" << prefill_fallback_ns.load(std::memory_order_relaxed) / 1000000
              << ", total_disk_io_ms=" << disk_io_ns.load(std::memory_order_relaxed) / 1000000
              << ", total_transpose_ms=" << transpose_ns.load(std::memory_order_relaxed) / 1000000
              << ", total_gpu_copy_ms=" << gpu_copy_ns.load(std::memory_order_relaxed) / 1000000
              << ", batched_fallbacks=" << batched_fallbacks.load(std::memory_order_relaxed)
              << ", grouped_fallbacks=" << grouped_fallbacks.load(std::memory_order_relaxed)
              << ", created_onednn_kernels=" << created_onednn_kernels.load(std::memory_order_relaxed) << std::endl;
}

OtdPerfCounters* get_perf_counters() {
    static bool enabled = std::getenv("MOE_OTD_PERF_LOG") != nullptr;
    if (!enabled) {
        return nullptr;
    }

    static OtdPerfCounters counters;
    static bool registered = [] {
        std::atexit([] {
            counters.dump();
        });
        return true;
    }();
    (void)registered;
    return &counters;
}

namespace {

class RouteTraceWriter {
public:
    RouteTraceWriter() {
        const char* path = std::getenv("MOE_OTD_TRACE");
        if (path && *path) {
            _output.open(path, std::ios::app);
            if (_output) {
                _output << "# step layer token experts\n";
            } else {
                std::cerr << "[OTD_TRACE] failed to open route trace file: " << path << std::endl;
            }
        }
    }

    void append(const std::string& primitive_id,
                size_t token_num,
                size_t top_k,
                const std::vector<uint32_t>& expert_ids) {
        if (!_output) {
            return;
        }

        std::lock_guard<std::mutex> lock(_mutex);
        auto [it, inserted] = _layer_indices.emplace(primitive_id, static_cast<uint32_t>(_layer_indices.size()));
        (void)inserted;
        if (it->second == 0) {
            if (_has_step) {
                ++_step;
            } else {
                _has_step = true;
            }
        }

        const auto layer = it->second;
        for (size_t t = 0; t < token_num; ++t) {
            _output << _step << ' ' << layer << ' ' << t;
            for (size_t k = 0; k < top_k; ++k) {
                _output << (k == 0 ? ' ' : ',') << expert_ids[t * top_k + k];
            }
            _output << '\n';
        }
        _output.flush();
        if (!_output) {
            std::cerr << "[OTD_TRACE] failed while writing route trace record" << std::endl;
            _output.close();
        }
    }

private:
    std::mutex _mutex;
    std::ofstream _output;
    std::unordered_map<std::string, uint32_t> _layer_indices;
    uint64_t _step = 0;
    bool _has_step = false;
};

}  // namespace

void record_route_trace(const std::string& primitive_id,
                        size_t token_num,
                        size_t top_k,
                        const std::vector<uint32_t>& expert_ids) {
    const char* path = std::getenv("MOE_OTD_TRACE");
    if (!path || !*path) {
        return;
    }
    if (top_k == 0 || token_num > std::numeric_limits<size_t>::max() / top_k || expert_ids.size() != token_num * top_k) {
        std::cerr << "[OTD_TRACE] invalid expert ID count; skipping route record" << std::endl;
        return;
    }
    static RouteTraceWriter writer;
    writer.append(primitive_id, token_num, top_k, expert_ids);
}

namespace {

bool needs_scale_zp_transpose(size_t offset_pos) {
    return (offset_pos >= 3 && offset_pos <= 8);
}

constexpr size_t transpose_tile_size = 32;

template <size_t ElementBytes>
void transpose_fixed_width_tiled(const uint8_t* src, uint8_t* dst, size_t oc, size_t group_count) {
    for (size_t o_block = 0; o_block < oc; o_block += transpose_tile_size) {
        const size_t o_end = std::min(o_block + transpose_tile_size, oc);
        for (size_t g_block = 0; g_block < group_count; g_block += transpose_tile_size) {
            const size_t g_end = std::min(g_block + transpose_tile_size, group_count);
            for (size_t o = o_block; o < o_end; ++o) {
                for (size_t g = g_block; g < g_end; ++g) {
                    const size_t src_offset = (o * group_count + g) * ElementBytes;
                    const size_t dst_offset = (g * oc + o) * ElementBytes;
                    if constexpr (ElementBytes == 1) {
                        dst[dst_offset] = src[src_offset];
                    } else {
                        std::memcpy(dst + dst_offset, src + src_offset, ElementBytes);
                    }
                }
            }
        }
    }
}

void transpose_dynamic_width_tiled(const uint8_t* src,
                                   uint8_t* dst,
                                   size_t oc,
                                   size_t group_count,
                                   size_t element_bytes) {
    for (size_t o_block = 0; o_block < oc; o_block += transpose_tile_size) {
        const size_t o_end = std::min(o_block + transpose_tile_size, oc);
        for (size_t g_block = 0; g_block < group_count; g_block += transpose_tile_size) {
            const size_t g_end = std::min(g_block + transpose_tile_size, group_count);
            for (size_t o = o_block; o < o_end; ++o) {
                for (size_t g = g_block; g < g_end; ++g) {
                    const size_t src_offset = (o * group_count + g) * element_bytes;
                    const size_t dst_offset = (g * oc + o) * element_bytes;
                    std::memcpy(dst + dst_offset, src + src_offset, element_bytes);
                }
            }
        }
    }
}

void transpose_by_element_width(const uint8_t* src,
                                uint8_t* dst,
                                size_t oc,
                                size_t group_count,
                                size_t element_bytes) {
    switch (element_bytes) {
    case 1:
        transpose_fixed_width_tiled<1>(src, dst, oc, group_count);
        break;
    case 2:
        transpose_fixed_width_tiled<2>(src, dst, oc, group_count);
        break;
    case 4:
        transpose_fixed_width_tiled<4>(src, dst, oc, group_count);
        break;
    default:
        transpose_dynamic_width_tiled(src, dst, oc, group_count, element_bytes);
        break;
    }
}

uint8_t get_u4(const uint8_t* packed, size_t index) {
    return static_cast<uint8_t>((packed[index / 2] >> ((index % 2) * 4)) & 0x0F);
}

void transpose_u4_zp_tiled(const uint8_t* src, uint8_t* dst, size_t oc, size_t group_count, size_t per_expert_size) {
    if (oc % 2 != 0) {
        std::memset(dst, 0, per_expert_size);
        for (size_t g = 0; g < group_count; ++g) {
            for (size_t o = 0; o < oc; ++o) {
                const size_t src_idx = o * group_count + g;
                const size_t dst_idx = g * oc + o;
                dst[dst_idx / 2] |= static_cast<uint8_t>(get_u4(src, src_idx) << ((dst_idx % 2) * 4));
            }
        }
        return;
    }

    for (size_t o_block = 0; o_block < oc; o_block += transpose_tile_size) {
        const size_t o_end = std::min(o_block + transpose_tile_size, oc) & ~size_t{1};
        for (size_t g_block = 0; g_block < group_count; g_block += transpose_tile_size) {
            const size_t g_end = std::min(g_block + transpose_tile_size, group_count);
            for (size_t g = g_block; g < g_end; ++g) {
                for (size_t o = o_block; o < o_end; o += 2) {
                    const size_t src_idx0 = o * group_count + g;
                    const size_t src_idx1 = (o + 1) * group_count + g;
                    const uint8_t lo = get_u4(src, src_idx0);
                    const uint8_t hi = get_u4(src, src_idx1);
                    const size_t dst_byte = (g * oc + o) / 2;
                    dst[dst_byte] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }
    }
}

void transpose_scale_zp_to_buffer(const cldnn::MOECompressed::Config& config,
                                  size_t offset_pos,
                                  const cldnn::layout& layout,
                                  const uint8_t* src,
                                  uint8_t* dst,
                                  size_t per_expert_size) {
    // Offset positions: 0-2 = weights (no transpose), 3-5 = scales, 6-8 = zero points.
    // Within each group of 3: index % 3 == 2 means "down" variant.
    const bool is_scale = (offset_pos >= 3 && offset_pos <= 5);
    const bool is_zp = (offset_pos >= 6 && offset_pos <= 8);
    if (!is_scale && !is_zp) {
        return;
    }

    const bool is_down = (offset_pos % 3 == 2);
    size_t oc = 0;
    size_t ic = 0;
    if (is_down) {
        oc = static_cast<size_t>(config.hidden_size);
        ic = static_cast<size_t>(config.inter_size);
    } else {
        oc = static_cast<size_t>(config.inter_size);
        ic = static_cast<size_t>(config.hidden_size);
    }

    const size_t group_size = static_cast<size_t>(config.group_size);
    size_t group_count = 1;
    if (group_size != 0 && group_size != std::numeric_limits<size_t>::max()) {
        OPENVINO_ASSERT(ic % group_size == 0, "Invalid group_size for OTD transpose: offset_pos=", offset_pos, ", ic=", ic, ", group_size=", group_size);
        group_count = ic / group_size;
    }

    OPENVINO_ASSERT(oc > 0 && group_count > 0, "Invalid dims for OTD transpose: offset_pos=", offset_pos, ", oc=", oc, ", group_count=", group_count);

    const size_t elem_count = oc * group_count;
    if (is_scale) {
        const size_t elem_size = static_cast<size_t>(cldnn::data_type_traits::size_of(layout.data_type));
        OPENVINO_ASSERT(elem_size > 0, "Invalid scale element size for offset_pos=", offset_pos);
        OPENVINO_ASSERT(elem_count * elem_size == per_expert_size,
                        "Unexpected scale payload size for offset_pos=",
                        offset_pos,
                        ", expected=",
                        elem_count * elem_size,
                        ", got=",
                        per_expert_size);

        transpose_by_element_width(src, dst, oc, group_count, elem_size);
        return;
    }

    // Byte-addressable ZP uses the same tiled transpose; packed u4 is handled below.
    const auto zp_et = ov::element::Type(layout.data_type);
    const size_t zp_bitwidth = zp_et.bitwidth();
    if (zp_bitwidth >= 8) {
        // u8 or larger: transpose like scale (1+ bytes per element)
        const size_t byte_per_elem = zp_bitwidth / 8;
        OPENVINO_ASSERT(elem_count * byte_per_elem == per_expert_size,
                        "Unexpected zp payload size for offset_pos=",
                        offset_pos,
                        ", expected=",
                        elem_count * byte_per_elem,
                        ", got=",
                        per_expert_size);

        transpose_by_element_width(src, dst, oc, group_count, byte_per_elem);
        return;
    }

    // u4 packed: 2 elements per byte
    OPENVINO_ASSERT(elem_count % 2 == 0, "Unexpected odd element count for packed zp offset_pos=", offset_pos, ", elem_count=", elem_count);
    OPENVINO_ASSERT(elem_count / 2 == per_expert_size,
                    "Unexpected zp payload size for offset_pos=",
                    offset_pos,
                    ", expected=",
                    elem_count / 2,
                    ", got=",
                    per_expert_size);

    transpose_u4_zp_tiled(src, dst, oc, group_count, per_expert_size);
}

}  // namespace

void maybe_transpose_scale_zp(const cldnn::MOECompressed::Config& config,
                              size_t offset_pos,
                              const cldnn::layout& layout,
                              std::vector<uint8_t>& payload,
                              size_t per_expert_size) {
    if (!needs_scale_zp_transpose(offset_pos)) {
        return;
    }
    std::vector<uint8_t> transposed(per_expert_size, 0);
    transpose_scale_zp_to_buffer(config, offset_pos, layout, payload.data(), transposed.data(), per_expert_size);
    payload.swap(transposed);
}

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         ParallelWeightReader& weight_reader,
                         cldnn::moe_weights& wei_mem,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts) {
    struct tensor_fill_plan {
        size_t per_expert_size = 0;
        size_t src_offset = 0;
        size_t dst_offset = 0;
    };

    const auto num_expert = static_cast<size_t>(config.num_expert);

    OPENVINO_ASSERT(weight_bin_offsets.size() == cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count, "Unexpected number of MOE weight offsets");

    static const std::array<const char*, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> tensor_names = {
        {"gate_w", "up_w", "down_w", "gate_s", "up_s", "down_s", "gate_z", "up_z", "down_z"}};
    const std::array<cldnn::memory_ptr, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> tensors_by_offset = {
        {wei_mem.gate_w, wei_mem.up_w, wei_mem.down_w, wei_mem.gate_s, wei_mem.up_s, wei_mem.down_s, wei_mem.gate_z, wei_mem.up_z, wei_mem.down_z}};

    auto* perf = get_perf_counters();

    const size_t weight_file_size = weight_reader.file_size();
    std::vector<uint8_t> payload;

    size_t index = 0;
    for (uint32_t expert : experts_list) {
        auto make_tensor_fill_plan = [&](size_t base_offset, cldnn::memory_ptr mem, size_t expert_no, size_t lru_expert_no, const char* tensor_name) {
            tensor_fill_plan plan;
            if (!mem) {
                return plan;
            }
            const auto total_bytes = mem->get_layout().bytes_count();
            OPENVINO_ASSERT(num_expert > 0, "Invalid expert count");
            plan.per_expert_size = total_bytes / num_expert;
            plan.src_offset = base_offset + expert_no * plan.per_expert_size;
            plan.dst_offset = lru_expert_no * plan.per_expert_size;
            OPENVINO_ASSERT(plan.src_offset <= weight_file_size, "Invalid src_offset out of file: ", plan.src_offset, ", file_size=", weight_file_size);
            OPENVINO_ASSERT(plan.per_expert_size <= weight_file_size - plan.src_offset,
                            "Read range out of file for tensor ",
                            tensor_name,
                            ": src_offset=",
                            plan.src_offset,
                            ", per_expert_size=",
                            plan.per_expert_size,
                            ", file_size=",
                            weight_file_size,
                            ", base_offset=",
                            base_offset,
                            ", expert=",
                            expert_no);
            return plan;
        };

        for (size_t offset_pos = 0; offset_pos < static_cast<size_t>(cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count); offset_pos++) {
            auto plan =
                make_tensor_fill_plan(weight_bin_offsets[offset_pos], tensors_by_offset[offset_pos], expert, lru_experts[index], tensor_names[offset_pos]);
            auto mem = tensors_by_offset[offset_pos];
            if (!mem || plan.per_expert_size == 0) {
                continue;
            }

            const bool direct_host_write = mem->get_allocation_type() == cldnn::allocation_type::usm_host;
            const bool transpose = needs_scale_zp_transpose(offset_pos);
            uint8_t* host_dst = nullptr;
            if (direct_host_write) {
                host_dst = static_cast<uint8_t*>(mem->buffer_ptr());
                OPENVINO_ASSERT(host_dst != nullptr, "OTD usm_host slot has no host-accessible pointer for ", tensor_names[offset_pos]);
                host_dst += plan.dst_offset;
            }

            // Raw routed weights can be read straight into a host-USM slot. Scale/ZP
            // tensors reuse this staging buffer because their packed data is transposed.
            uint8_t* read_buffer = host_dst;
            if (!direct_host_write || transpose) {
                if (payload.size() < plan.per_expert_size) {
                    payload.resize(plan.per_expert_size);
                }
                read_buffer = payload.data();
            }

            auto read_weights = [&] {
                weight_reader.read(reinterpret_cast<char*>(read_buffer), plan.per_expert_size, plan.src_offset);
            };
            if (perf) {
                const auto t0 = std::chrono::steady_clock::now();
                read_weights();
                const auto t1 = std::chrono::steady_clock::now();
                perf->disk_io_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                                           std::memory_order_relaxed);
            } else {
                read_weights();
            }

            if (direct_host_write) {
                if (transpose) {
                    if (perf) {
                        const auto t0 = std::chrono::steady_clock::now();
                        transpose_scale_zp_to_buffer(config, offset_pos, mem->get_layout(), payload.data(), host_dst, plan.per_expert_size);
                        const auto t1 = std::chrono::steady_clock::now();
                        perf->transpose_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                                                     std::memory_order_relaxed);
                    } else {
                        transpose_scale_zp_to_buffer(config, offset_pos, mem->get_layout(), payload.data(), host_dst, plan.per_expert_size);
                    }
                }
                if (perf) {
                    perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
                }
                continue;
            }

            // Device-USM fallback retains the existing staged transpose and blocking upload.
            if (perf) {
                const auto t0 = std::chrono::steady_clock::now();
                maybe_transpose_scale_zp(config, offset_pos, mem->get_layout(), payload, plan.per_expert_size);
                const auto t1 = std::chrono::steady_clock::now();
                mem->copy_from(exec_stream, payload.data(), 0, plan.dst_offset, plan.per_expert_size, true);
                const auto t2 = std::chrono::steady_clock::now();
                perf->transpose_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                                             std::memory_order_relaxed);
                perf->gpu_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()),
                                            std::memory_order_relaxed);
                perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                maybe_transpose_scale_zp(config, offset_pos, mem->get_layout(), payload, plan.per_expert_size);
                mem->copy_from(exec_stream, payload.data(), 0, plan.dst_offset, plan.per_expert_size, true);
            }
        }

        index++;
    }
}

}  // namespace ov::intel_gpu::ocl::moe_otd
