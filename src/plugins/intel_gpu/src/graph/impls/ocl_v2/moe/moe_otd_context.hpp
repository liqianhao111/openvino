// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "expert_weight_provider.hpp"

namespace ov::intel_gpu::ocl::moe {

// Per-network registry of MoE expert providers by layer index, so layer L can prefetch into layer L+1.
class MoeOtdContext {
public:
    // Shared by all MoE impls that pass the same key (their network); released with the last holder.
    static std::shared_ptr<MoeOtdContext> get(const void* key);

    void register_layer(int64_t layer, const std::shared_ptr<IExpertWeightProvider>& provider);
    std::shared_ptr<IExpertWeightProvider> layer(int64_t layer) const;

private:
    mutable std::mutex _mutex;
    std::unordered_map<int64_t, std::weak_ptr<IExpertWeightProvider>> _layers;
};

}  // namespace ov::intel_gpu::ocl::moe
