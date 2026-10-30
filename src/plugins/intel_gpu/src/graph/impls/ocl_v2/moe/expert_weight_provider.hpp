// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl::moe {

/// Slot indices produced by a simultaneous acquisition (parallel to the input experts vector).
using ExpertSlotLease = std::vector<size_t>;

// In-flight expert loads; defined by the offload provider.
struct PendingExpertLoad;

/// Result of a non-blocking acquisition. Slots stay pinned until IExpertWeightProvider::release().
struct ExpertRequest {
    ExpertSlotLease slots;                                    // parallel to the requested experts
    std::vector<uint8_t> ready;                               // parallel; 1 if the slot held the weights at request time
    std::vector<size_t> pinned;                               // unique pinned slots
    std::vector<std::shared_ptr<PendingExpertLoad>> pending;  // loads that must finish before the not-ready slots are valid

    bool all_ready() const {
        return pending.empty();
    }
};

/// Lightweight view of routed expert weight/scale/zp triplets for 3 GEMMs.
struct RoutedWeightViews {
    cldnn::memory::ptr weight[3];
    cldnn::memory::ptr scale[3];
    cldnn::memory::ptr zp[3];
};

// Abstraction over "where the weights of a MoE layer's experts live" and how they
// are made addressable on the device for a given set of activated experts.
//
// Two concrete strategies exist today, both expressible through this interface:
//   * Fully resident: every expert's weights stay in device memory for the whole
//     network lifetime (the ratio=0 path). acquire() is the identity mapping.
//   * Offloaded (OTD): expert weights live on disk and are streamed on demand into
//     a bounded set of device-resident slots managed by an LRU cache (ratio>0).
//     acquire() deduplicates, assigns/streams slots, and returns the slot per expert.
//
// Keeping this behind an interface lets the MoE execution paths be written once,
// independent of the residency strategy, and lets the offload machinery (LRU +
// disk streaming + capacity serialization) be reused by other plugins/back-ends
// by only swapping the device-buffer/stream adapter underneath.
class IExpertWeightProvider {
public:
    virtual ~IExpertWeightProvider() = default;

    // Number of device-resident expert slots. 0 means fully resident (no bound on
    // how many experts can be addressed at once).
    virtual size_t resident_capacity() const = 0;

    // True when expert weights are streamed on demand rather than fully resident.
    virtual bool is_offloaded() const = 0;

    // Fills the RoutedWeightViews with the correct memory pointers for the 3 GEMMs.
    virtual void fill_routed_weight_views(cldnn::moe_weights& weights, RoutedWeightViews& views) = 0;

    // Non-blocking: pins a slot per expert and starts loading the missing ones.
    // Returns nullopt if the unique experts exceed the slots that can be assigned.
    virtual std::optional<ExpertRequest> request(const std::vector<uint32_t>& experts, cldnn::stream& stream) = 0;

    // Blocks until every slot of the request holds its expert. Rethrows load errors.
    virtual void wait(ExpertRequest& request) = 0;

    // Unpins the request's slots. Kernels reading them must be enqueued before the next request
    // that may evict them is issued.
    virtual void release(ExpertRequest& request) = 0;

    // Best effort, non-blocking: starts low-priority loads of experts a later request is likely to need.
    // Must not be called while kernels that read this provider's slots may still be running.
    virtual void prefetch(const std::vector<uint32_t>& /*experts*/, cldnn::stream& /*stream*/) {}

    // Blocking acquisition: request + wait + release. The lease is parallel to the input experts vector.
    std::optional<ExpertSlotLease> try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
        auto req = request(experts, stream);
        if (!req) {
            return std::nullopt;
        }
        try {
            wait(*req);
        } catch (...) {
            release(*req);
            throw;
        }
        release(*req);
        return std::move(req->slots);
    }

    size_t acquire_one(uint32_t expert, cldnn::stream& stream) {
        auto lease = try_acquire_simultaneous({expert}, stream);
        OPENVINO_ASSERT(lease.has_value(), "No evictable MoE OTD slot for expert ", expert);
        return (*lease)[0];
    }

    // Binds device-resident weight buffers. Called once on first execution.
    virtual void bind(cldnn::moe_weights& resident) = 0;

    // Returns true after bind() has been called.
    virtual bool is_bound() const = 0;
};

}  // namespace ov::intel_gpu::ocl::moe
