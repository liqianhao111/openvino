// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in.
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

struct PendingExpertLoad {
    std::shared_ptr<moe_otd::MoeIoEngine> engine;
    std::shared_ptr<ExpertSlotCache> cache;
    cldnn::MOECompressed::Config config{};
    std::filesystem::path weights_path;
    std::vector<moe_otd::HostTensorLoad> loads;
    std::vector<size_t> slots;  // per expert
    std::vector<uint64_t> generations;  // per expert; the slot assignment this load fills
    std::unique_ptr<std::atomic<size_t>[]> remaining_tasks;  // per expert
    moe_otd::MoeIoEngine::Ticket ticket;

    std::mutex mutex;
    bool finished = false;
    std::exception_ptr error;

    // Waits for the IO batch and drops the mapping of every slot whose load did not complete.
    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if (finished) {
            return;
        }
        try {
            if (ticket) {
                engine->wait(ticket);
            }
        } catch (...) {
            error = std::current_exception();
        }
        for (size_t i = 0; i < slots.size(); ++i) {
            const size_t left = remaining_tasks ? remaining_tasks[i].load(std::memory_order_acquire) : 1;
            if (left == 0) {
                cache->mark_ready(slots[i], generations[i]);
            } else {
                cache->mark_failed(slots[i], generations[i]);
            }
        }
        finished = true;
    }

    ~PendingExpertLoad() {
        // IO tasks reference this object.
        finish();
    }
};

std::optional<ExpertRequest> ResidentExpertWeightProvider::request(const std::vector<uint32_t>& experts, cldnn::stream& /*stream*/) {
    ExpertRequest req;
    req.slots.assign(experts.begin(), experts.end());
    req.ready.assign(experts.size(), 1);
    return req;
}

OffloadExpertWeightProvider::OffloadExpertWeightProvider(size_t capacity,
                                                         const cldnn::MOECompressed::Config& config,
                                                         std::vector<size_t> weight_bin_offsets,
                                                         std::filesystem::path weights_path)
    : _capacity(capacity),
      _config(config),
      _weight_bin_offsets(std::move(weight_bin_offsets)),
      _weights_path(std::move(weights_path)),
      _cache(std::make_shared<ExpertSlotCache>(capacity)),
      _weight_reader(_weights_path),
      _slot_pending(capacity) {}

void OffloadExpertWeightProvider::bind(cldnn::moe_weights& resident) {
    _resident = &resident;
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

std::optional<ExpertRequest> OffloadExpertWeightProvider::request(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    // Reject oversized batches before touching the cache or reading any weights.
    std::unordered_map<uint32_t, size_t> unique_index;
    std::vector<uint32_t> unique_experts;
    unique_index.reserve(experts.size());
    for (const uint32_t expert : experts) {
        if (unique_index.emplace(expert, unique_experts.size()).second) {
            unique_experts.push_back(expert);
            if (unique_experts.size() > _capacity) {
                return std::nullopt;
            }
        }
    }

    std::lock_guard<std::mutex> lock(_request_mutex);
    auto acquired = _cache->acquire(unique_experts);
    if (!acquired) {
        return std::nullopt;
    }

    ExpertRequest req;
    std::vector<uint8_t> unique_ready(unique_experts.size(), 0);
    std::vector<uint32_t> missing_experts;
    std::vector<size_t> missing_slots;
    std::vector<uint64_t> missing_generations;
    std::vector<size_t> missing_unique;
    req.pinned.reserve(unique_experts.size());
    for (size_t u = 0; u < unique_experts.size(); ++u) {
        const auto& a = (*acquired)[u];
        req.pinned.push_back(a.slot);
        if (a.state == SlotState::ready) {
            unique_ready[u] = 1;
        } else if (a.state == SlotState::loading) {
            const auto& owner = _slot_pending[a.slot];
            OPENVINO_ASSERT(owner != nullptr, "MoE OTD slot ", a.slot, " is LOADING without an owner");
            if (std::find(req.pending.begin(), req.pending.end(), owner) == req.pending.end()) {
                req.pending.push_back(owner);
            }
        } else {
            missing_experts.push_back(unique_experts[u]);
            missing_slots.push_back(a.slot);
            missing_generations.push_back(a.generation);
            missing_unique.push_back(u);
        }
    }

    if (auto* perf = moe_otd::get_perf_counters()) {
        perf->gpu_hits.fetch_add(unique_experts.size() - missing_experts.size(), std::memory_order_relaxed);
        perf->gpu_misses.fetch_add(missing_experts.size(), std::memory_order_relaxed);
        for (const auto& a : *acquired) {
            perf->prefetch_hits.fetch_add(a.prefetch_hit ? 1 : 0, std::memory_order_relaxed);
            perf->prefetch_wasted.fetch_add(a.evicted_unused_prefetch ? 1 : 0, std::memory_order_relaxed);
        }
    }

    if (!missing_experts.empty()) {
        try {
            auto pending = start_load(missing_experts, missing_slots, missing_generations, stream);
            if (pending) {
                for (const size_t slot : missing_slots) {
                    _slot_pending[slot] = pending;
                }
                req.pending.push_back(std::move(pending));
            } else {
                for (size_t i = 0; i < missing_slots.size(); ++i) {
                    _cache->mark_ready(missing_slots[i], missing_generations[i]);
                    unique_ready[missing_unique[i]] = 1;
                }
            }
        } catch (...) {
            for (size_t i = 0; i < missing_slots.size(); ++i) {
                _cache->mark_failed(missing_slots[i], missing_generations[i]);
            }
            for (const size_t slot : req.pinned) {
                _cache->unpin(slot);
            }
            throw;
        }
    }

    req.slots.resize(experts.size());
    req.ready.resize(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) {
        const size_t u = unique_index[experts[i]];
        req.slots[i] = (*acquired)[u].slot;
        req.ready[i] = unique_ready[u];
    }
    return req;
}

void OffloadExpertWeightProvider::wait(ExpertRequest& req) {
    if (req.pending.empty()) {
        return;
    }
    auto* perf = moe_otd::get_perf_counters();
    {
        moe_otd::ScopedOtdTimer wait_timer(perf ? &perf->io_wait_ns : nullptr);
        for (const auto& pending : req.pending) {
            pending->finish();
        }
    }

    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(_request_mutex);
        for (const auto& pending : req.pending) {
            if (pending->error && !error) {
                error = pending->error;
            }
            for (const size_t slot : pending->slots) {
                if (_slot_pending[slot] == pending) {
                    _slot_pending[slot].reset();
                }
            }
        }
    }
    req.pending.clear();
    if (error) {
        std::rethrow_exception(error);
    }
}

void OffloadExpertWeightProvider::release(ExpertRequest& req) {
    for (const size_t slot : req.pinned) {
        _cache->unpin(slot);
    }
    req.pinned.clear();
}

void OffloadExpertWeightProvider::prefetch(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    if (!_bound || !supports_async_load()) {
        return;
    }
    auto* perf = moe_otd::get_perf_counters();
    std::lock_guard<std::mutex> lock(_request_mutex);
    for (const uint32_t expert : experts) {
        if (_cache->find(expert)) {
            continue;
        }
        auto acquired = _cache->acquire({expert}, /*prefetch=*/true);
        if (!acquired) {
            return;
        }
        const auto& a = (*acquired)[0];
        // A LOADING slot is never evicted, so the pin is not needed past acquire().
        _cache->unpin(a.slot);
        try {
            // One batch per expert: a request that needs it then only runs this expert's unstarted reads.
            _slot_pending[a.slot] = start_load({expert}, {a.slot}, {a.generation}, stream, moe_otd::IoPriority::prefetch);
        } catch (...) {
            _cache->mark_failed(a.slot, a.generation);
            throw;
        }
        if (perf) {
            perf->prefetch_issued.fetch_add(1, std::memory_order_relaxed);
            perf->prefetch_wasted.fetch_add(a.evicted_unused_prefetch ? 1 : 0, std::memory_order_relaxed);
        }
    }
}

bool OffloadExpertWeightProvider::supports_async_load() {
    std::call_once(_io_engine_once, [this] {
        _io_engine = moe_otd::MoeIoEngine::shared();
    });
    return _io_engine && _resident && moe_otd::supports_direct_host_load(*_resident);
}

std::shared_ptr<PendingExpertLoad> OffloadExpertWeightProvider::start_load(const std::vector<uint32_t>& experts,
                                                                           const std::vector<size_t>& slots,
                                                                           const std::vector<uint64_t>& generations,
                                                                           cldnn::stream& stream,
                                                                           moe_otd::IoPriority priority) {
    OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
    if (!supports_async_load()) {
        moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, experts, slots);
        return nullptr;
    }

    auto pending = std::make_shared<PendingExpertLoad>();
    pending->engine = _io_engine;
    pending->cache = _cache;
    pending->config = _config;
    pending->weights_path = _weights_path;
    pending->slots = slots;
    pending->generations = generations;
    pending->remaining_tasks = std::make_unique<std::atomic<size_t>[]>(experts.size());

    const size_t weight_file_size = _weight_reader.file_size();
    std::vector<size_t> load_expert;  // expert index of each load
    for (size_t i = 0; i < experts.size(); i++) {
        const size_t first = pending->loads.size();
        moe_otd::plan_direct_host_loads(_config,
                                        _weight_bin_offsets,
                                        weight_file_size,
                                        *_resident,
                                        experts[i],
                                        slots[i],
                                        moe_otd::direct_read_chunk_bytes,
                                        pending->loads);
        pending->remaining_tasks[i].store(pending->loads.size() - first, std::memory_order_relaxed);
        load_expert.resize(pending->loads.size(), i);
    }

    std::vector<moe_otd::MoeIoEngine::Task> tasks;
    tasks.reserve(pending->loads.size());
    for (size_t l = 0; l < pending->loads.size(); ++l) {
        tasks.emplace_back([p = pending.get(), l, e = load_expert[l]](moe_otd::IoWorkerContext& context) {
            moe_otd::execute_direct_host_load(p->config, p->loads[l], p->weights_path, context);
            if (p->remaining_tasks[e].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                p->cache->mark_ready(p->slots[e], p->generations[e]);
            }
        });
    }

    if (auto* perf = moe_otd::get_perf_counters()) {
        perf->io_batches.fetch_add(1, std::memory_order_relaxed);
        perf->io_tasks.fetch_add(tasks.size(), std::memory_order_relaxed);
    }
    pending->ticket = _io_engine->submit(std::move(tasks), priority);
    return pending;
}

}  // namespace ov::intel_gpu::ocl::moe
