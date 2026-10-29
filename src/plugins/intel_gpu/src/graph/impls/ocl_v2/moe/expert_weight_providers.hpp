// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "expert_slot_cache.hpp"
#include "expert_weight_provider.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

// Identity provider: every expert's weights are already resident in device memory
// for the network lifetime. This is the behaviour of the non-offloaded (ratio=0) path.
class ResidentExpertWeightProvider : public IExpertWeightProvider {
public:
    size_t resident_capacity() const override {
        return 0;
    }
    bool is_offloaded() const override {
        return false;
    }
    void bind(cldnn::moe_weights& /*resident*/) override {}
    bool is_bound() const override {
        return true;
    }
    void fill_routed_weight_views(cldnn::moe_weights& weights, RoutedWeightViews& views) override {
        views.weight[0] = weights.gate_w;
        views.scale[0] = weights.gate_s;
        views.zp[0] = weights.gate_z;
        views.weight[1] = weights.up_w;
        views.scale[1] = weights.up_s;
        views.zp[1] = weights.up_z;
        views.weight[2] = weights.down_w;
        views.scale[2] = weights.down_s;
        views.zp[2] = weights.down_z;
    }

    std::optional<ExpertRequest> request(const std::vector<uint32_t>& experts, cldnn::stream& stream) override;
    void wait(ExpertRequest& /*request*/) override {}
    void release(ExpertRequest& /*request*/) override {}
};

// Offload (OTD) provider: expert weights live on disk and are streamed on demand
// into a bounded set of device-resident slots managed by an ExpertSlotCache.
// The provider deduplicates the requested experts, assigns/streams slots, and
// returns the slot index per requested expert.
//
// The resident device buffers are not owned here; they are bound at runtime via
// bind() because in this back-end they alias the primitive's
// input memory. Loading metadata (config / bin offsets / weights path) is captured
// at construction from graph-build time information.
class OffloadExpertWeightProvider : public IExpertWeightProvider {
public:
    OffloadExpertWeightProvider(size_t capacity,
                                const cldnn::MOECompressed::Config& config,
                                std::vector<size_t> weight_bin_offsets,
                                std::filesystem::path weights_path);

    size_t resident_capacity() const override {
        return _capacity;
    }
    bool is_offloaded() const override {
        return true;
    }

    void bind(cldnn::moe_weights& resident) override;
    bool is_bound() const override {
        return _bound;
    }
    void fill_routed_weight_views(cldnn::moe_weights& weights, RoutedWeightViews& views) override;
    std::optional<ExpertRequest> request(const std::vector<uint32_t>& experts, cldnn::stream& stream) override;
    void wait(ExpertRequest& request) override;
    void release(ExpertRequest& request) override;

    ExpertSlotCache& cache() {
        return *_cache;
    }

    moe_otd::ParallelWeightReader& weight_reader() {
        return _weight_reader;
    }

private:
    // Starts loading (expert, slot) misses. Host-USM slots load on the IO engine and return the
    // in-flight handle; otherwise they load serially through the GPU queue and nullptr is returned.
    std::shared_ptr<PendingExpertLoad> start_load(const std::vector<uint32_t>& experts, const std::vector<size_t>& slots, cldnn::stream& stream);

    size_t _capacity = 0;
    cldnn::MOECompressed::Config _config{};
    std::vector<size_t> _weight_bin_offsets;
    std::filesystem::path _weights_path;
    std::shared_ptr<ExpertSlotCache> _cache;
    moe_otd::ParallelWeightReader _weight_reader;
    std::shared_ptr<moe_otd::MoeIoEngine> _io_engine;
    std::once_flag _io_engine_once;
    // Serializes request(); _slot_pending[s] is the load that owns slot s while it is LOADING.
    std::mutex _request_mutex;
    std::vector<std::shared_ptr<PendingExpertLoad>> _slot_pending;
    cldnn::moe_weights* _resident = nullptr;
    bool _bound = false;
};

}  // namespace ov::intel_gpu::ocl::moe
