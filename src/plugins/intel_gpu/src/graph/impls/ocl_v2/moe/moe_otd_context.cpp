// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_context.hpp"

namespace ov::intel_gpu::ocl::moe {

std::shared_ptr<MoeOtdContext> MoeOtdContext::get(const void* key) {
    static std::mutex mutex;
    static std::unordered_map<const void*, std::weak_ptr<MoeOtdContext>> contexts;

    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = contexts.begin(); it != contexts.end();) {
        it = it->second.expired() ? contexts.erase(it) : std::next(it);
    }
    auto& entry = contexts[key];
    auto context = entry.lock();
    if (!context) {
        context = std::make_shared<MoeOtdContext>();
        entry = context;
    }
    return context;
}

void MoeOtdContext::register_layer(int64_t layer, const std::shared_ptr<IExpertWeightProvider>& provider) {
    std::lock_guard<std::mutex> lock(_mutex);
    _layers[layer] = provider;
}

std::shared_ptr<IExpertWeightProvider> MoeOtdContext::layer(int64_t layer) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _layers.find(layer);
    return it == _layers.end() ? nullptr : it->second.lock();
}

}  // namespace ov::intel_gpu::ocl::moe
