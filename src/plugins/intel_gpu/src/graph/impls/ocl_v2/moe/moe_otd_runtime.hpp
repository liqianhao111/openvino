// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "lru_cache.hpp"
#include "moe_otd_io.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

// Lightweight perf counters for OTD profiling.
// Enabled by setting MOE_OTD_PERF_LOG=1 environment variable.
// Counters are printed to stderr on process exit.
struct OtdPerfCounters {
    std::atomic<uint64_t> moe_executions{0};
    std::atomic<uint64_t> gpu_hits{0};
    std::atomic<uint64_t> gpu_misses{0};
    // disk_io_ns / transpose_ns are summed over IO worker threads, not wall time.
    std::atomic<uint64_t> disk_io_ns{0};
    std::atomic<uint64_t> transpose_ns{0};
    std::atomic<uint64_t> gpu_copy_ns{0};
    std::atomic<uint64_t> io_batches{0};
    std::atomic<uint64_t> io_tasks{0};
    std::atomic<uint64_t> io_wait_ns{0};
    std::atomic<uint64_t> finish_wait_ns{0};
    std::atomic<uint64_t> topk_readback_ns{0};
    std::atomic<uint64_t> index_write_ns{0};
    std::atomic<uint64_t> batched_gemv_prepare_ns{0};
    std::atomic<uint64_t> tensor_load_count{0};
    std::atomic<uint64_t> batched_fallbacks{0};
    std::atomic<uint64_t> batched_split_passes{0};
    // IO wait after the ready experts were dispatched in a split batched GEMV.
    std::atomic<uint64_t> split_wait_ns{0};
    std::atomic<uint64_t> prefetch_issued{0};
    // Demand hits on prefetched experts (loaded or still loading) / prefetched experts evicted unused.
    std::atomic<uint64_t> prefetch_hits{0};
    std::atomic<uint64_t> prefetch_wasted{0};
    std::atomic<uint64_t> hint_readback_ns{0};
    std::atomic<uint64_t> grouped_fallbacks{0};
    std::atomic<uint64_t> prefill_fallbacks{0};
    std::atomic<uint64_t> prefill_fallback_ns{0};
    std::atomic<uint64_t> created_onednn_kernels{0};

    void dump() const;
};

OtdPerfCounters* get_perf_counters();

class ScopedOtdTimer {
public:
    explicit ScopedOtdTimer(std::atomic<uint64_t>* counter) : _counter(counter) {
        if (_counter) {
            _start = std::chrono::steady_clock::now();
        }
    }

    ~ScopedOtdTimer() {
        if (_counter) {
            const auto elapsed = std::chrono::steady_clock::now() - _start;
            _counter->fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                                std::memory_order_relaxed);
        }
    }

    ScopedOtdTimer(const ScopedOtdTimer&) = delete;
    ScopedOtdTimer& operator=(const ScopedOtdTimer&) = delete;

private:
    std::atomic<uint64_t>* _counter;
    std::chrono::steady_clock::time_point _start{};
};

// When MOE_OTD_TRACE is set, appends one text line per token: "step layer token e0,e1,...".
// layer is assigned in first-execution order; step advances each time layer 0 runs again.
void record_route_trace(const std::string& primitive_id,
                        size_t token_num,
                        size_t top_k,
                        const std::vector<uint32_t>& expert_ids);

class ParallelWeightReader {
public:
    explicit ParallelWeightReader(const std::filesystem::path& weights_path) : _weights_path(weights_path) {}

    ~ParallelWeightReader() {
        if (_opened) {
            ov::util::close_file_handle(_shared_handle);
        }
    }

    ParallelWeightReader(const ParallelWeightReader&) = delete;
    ParallelWeightReader& operator=(const ParallelWeightReader&) = delete;

    const std::filesystem::path& path() const {
        return _weights_path;
    }

    size_t file_size() {
        ensure_open();
        return _file_size;
    }

    void read(char* dst, size_t size, size_t file_offset) {
        ensure_open();
        if (!ov::util::positional_read(_shared_handle, dst, size, file_offset)) {
            OPENVINO_THROW("Failed to read enough bytes from OTD weight file");
        }
    }

private:
    void ensure_open() {
        if (!_opened) {
            std::streamoff file_size = 0;
            ov::util::get_file_handle_and_size(_weights_path, 0, _shared_handle, file_size);
            _file_size = static_cast<size_t>(file_size);
            _opened = true;
        }
    }

    std::filesystem::path _weights_path;
    ov::FileHandle _shared_handle{};
    size_t _file_size = 0;
    bool _opened = false;
};

void maybe_transpose_scale_zp(const cldnn::MOECompressed::Config& config,
                              size_t offset_pos,
                              const cldnn::layout& layout,
                              std::vector<uint8_t>& payload,
                              size_t per_expert_size);

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         ParallelWeightReader& weight_reader,
                         cldnn::moe_weights& wei_mem,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts);

// Raw (non-transposed) expert tensors are split into reads of at most this size so that a
// single missing expert still spreads over several IO workers.
inline constexpr size_t direct_read_chunk_bytes = 128 * 1024;

// One IO task: reads an expert tensor (or a chunk of a raw weight tensor) straight into its
// host-USM slot, transposing scale/zp through the worker staging buffer.
struct HostTensorLoad {
    size_t offset_pos = 0;
    size_t file_offset = 0;
    size_t size = 0;
    uint8_t* dst = nullptr;
    cldnn::data_types data_type = cldnn::data_types::dynamic;
    bool counts_as_tensor = true;
};

// True when every non-empty routed slot tensor is host-USM, so loads can bypass the GPU queue
// and run on IO worker threads.
bool supports_direct_host_load(const cldnn::moe_weights& wei_mem);

void plan_direct_host_loads(const cldnn::MOECompressed::Config& config,
                            const std::vector<size_t>& weight_bin_offsets,
                            size_t weight_file_size,
                            const cldnn::moe_weights& wei_mem,
                            uint32_t expert,
                            size_t slot,
                            size_t max_raw_chunk_bytes,
                            std::vector<HostTensorLoad>& loads);

void execute_direct_host_load(const cldnn::MOECompressed::Config& config,
                              const HostTensorLoad& load,
                              const std::filesystem::path& weights_path,
                              IoWorkerContext& context);

}  // namespace ov::intel_gpu::ocl::moe_otd