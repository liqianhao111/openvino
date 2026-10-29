// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in.
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

std::optional<ExpertSlotLease> ResidentExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& /*stream*/) {
    // Resident: identity mapping, always succeeds.
    return std::vector<size_t>(experts.begin(), experts.end());
}

size_t ResidentExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& /*stream*/) {
    return expert;
}

OffloadExpertWeightProvider::OffloadExpertWeightProvider(size_t capacity,
                                                         const cldnn::MOECompressed::Config& config,
                                                         std::vector<size_t> weight_bin_offsets,
                                                         std::filesystem::path weights_path)
    : _capacity(capacity),
      _config(config),
      _weight_bin_offsets(std::move(weight_bin_offsets)),
      _weights_path(std::move(weights_path)),
      _cache(std::make_shared<LRUCache>(capacity)),
      _weight_reader(_weights_path) {}

void OffloadExpertWeightProvider::bind(cldnn::moe_weights& resident) {
    _resident = &resident;
    _cache->set_initialized();
    _bound = true;
    GPU_DEBUG_TRACE << "MOE OTD bind: resident_slots=" << _capacity << " total_experts=" << _config.num_expert
                    << " shared_expert=" << (_config.num_shared_expert > 0 ? 1 : 0) << std::endl;
}

void OffloadExpertWeightProvider::fill_routed_weight_views(cldnn::moe_weights& /*weights*/, RoutedWeightViews& views) {
    // Offloaded: point at the LRU resident buffers (not the original weight inputs).
    OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
    views.weight[0] = _resident->gate_w;
    views.scale[0] = _resident->gate_s;
    views.zp[0] = _resident->gate_z;
    views.weight[1] = _resident->up_w;
    views.scale[1] = _resident->up_s;
    views.zp[1] = _resident->up_z;
    views.weight[2] = _resident->down_w;
    views.scale[2] = _resident->down_s;
    views.zp[2] = _resident->down_z;
}

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    // Reject oversized batches before touching the LRU or reading any weights.
    std::unordered_set<uint32_t> unique_experts;
    unique_experts.reserve(experts.size());
    for (const uint32_t expert : experts) {
        unique_experts.insert(expert);
        if (unique_experts.size() > _capacity) {
            return std::nullopt;
        }
    }

    std::unordered_map<uint32_t, size_t> expert_to_slot;
    expert_to_slot.reserve(unique_experts.size());

    std::vector<size_t> slots(experts.size());
    std::vector<uint32_t> missing_experts;
    std::vector<size_t> missing_slots;
    auto* perf = moe_otd::get_perf_counters();

    for (size_t i = 0; i < experts.size(); i++) {
        const uint32_t expert = experts[i];
        auto it = expert_to_slot.find(expert);
        if (it != expert_to_slot.end()) {
            slots[i] = it->second;
            continue;
        }

        // Unique experts <= capacity, so a later miss in this batch never evicts a slot assigned earlier.
        const auto item = _cache->get_lru_item(expert);
        const auto slot = item.first;

        if (item.second) {
            if (perf) {
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            if (perf) {
                perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
            }
            missing_experts.push_back(expert);
            missing_slots.push_back(slot);
        }

        expert_to_slot.emplace(expert, slot);
        slots[i] = slot;
    }

    if (!missing_experts.empty()) {
        load_experts(missing_experts, missing_slots, stream);
        for (const size_t slot : missing_slots) {
            _cache->set_filled(slot);
        }
    }

    return slots;
}

size_t OffloadExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& stream) {
    auto* perf = moe_otd::get_perf_counters();

    const auto item = _cache->get_lru_item(expert);
    const auto slot = item.first;

    if (item.second) {
        if (perf) {
            perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (perf) {
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        load_experts({expert}, {slot}, stream);
        _cache->set_filled(slot);
    }

    return slot;
}

void OffloadExpertWeightProvider::load_experts(const std::vector<uint32_t>& experts, const std::vector<size_t>& slots, cldnn::stream& stream) {
    OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
    std::call_once(_io_engine_once, [this] {
        _io_engine = moe_otd::MoeIoEngine::shared();
    });
    if (!_io_engine || !moe_otd::supports_direct_host_load(*_resident)) {
        moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, experts, slots);
        return;
    }

    const size_t weight_file_size = _weight_reader.file_size();
    std::vector<moe_otd::HostTensorLoad> loads;
    for (size_t i = 0; i < experts.size(); i++) {
        moe_otd::plan_direct_host_loads(_config,
                                        _weight_bin_offsets,
                                        weight_file_size,
                                        *_resident,
                                        experts[i],
                                        slots[i],
                                        moe_otd::direct_read_chunk_bytes,
                                        loads);
    }

    std::vector<moe_otd::MoeIoEngine::Task> tasks;
    tasks.reserve(loads.size());
    for (const auto& load : loads) {
        tasks.emplace_back([this, &load](moe_otd::IoWorkerContext& context) {
            moe_otd::execute_direct_host_load(_config, load, _weights_path, context);
        });
    }

    auto* perf = moe_otd::get_perf_counters();
    if (perf) {
        perf->io_batches.fetch_add(1, std::memory_order_relaxed);
        perf->io_tasks.fetch_add(tasks.size(), std::memory_order_relaxed);
    }
    moe_otd::ScopedOtdTimer wait_timer(perf ? &perf->io_wait_ns : nullptr);
    _io_engine->run(std::move(tasks));
}

}  // namespace ov::intel_gpu::ocl::moe
