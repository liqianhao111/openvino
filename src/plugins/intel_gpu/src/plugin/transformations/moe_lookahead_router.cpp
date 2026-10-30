// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_lookahead_router.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "intel_gpu/op/moe_router_fused.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu {

namespace {

struct MoELayer {
    std::shared_ptr<ov::op::internal::MOECompressed> moe;
    std::shared_ptr<ov::intel_gpu::op::MoERouterFused> router;
    std::shared_ptr<ov::op::v0::MatMul> router_matmul;
};

// Clones the non-constant weight decompression subgraph so the hint MatMul gets its own FC weights path;
// Constants stay shared.
ov::Output<ov::Node> clone_weight_subgraph(const ov::Output<ov::Node>& output,
                                           std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& cloned) {
    auto node = output.get_node_shared_ptr();
    if (ov::is_type<ov::op::v0::Constant>(node)) {
        return output;
    }
    auto it = cloned.find(node.get());
    if (it == cloned.end()) {
        ov::OutputVector inputs;
        for (const auto& input : node->input_values()) {
            inputs.push_back(clone_weight_subgraph(input, cloned));
        }
        auto copy = node->clone_with_new_inputs(inputs);
        copy->get_rt_info() = node->get_rt_info();
        copy->set_friendly_name(node->get_friendly_name() + "/lookahead");
        it = cloned.emplace(node.get(), copy).first;
    }
    return it->second->output(output.get_index());
}

}  // namespace

size_t MoELookaheadRouter::configured_lookahead_k() {
    const char* env = std::getenv("GPU_MOE_OTD_LOOKAHEAD_K");
    if (!env || !*env) {
        return use_router_top_k;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 10);
    return (end != env && *end == '\0') ? static_cast<size_t>(value) : use_router_top_k;
}

bool MoELookaheadRouter::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<MoELayer> layers;
    for (const auto& node : model->get_ordered_ops()) {
        auto moe = ov::as_type_ptr<ov::op::internal::MOECompressed>(node);
        if (!moe || moe->get_config().expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
            continue;
        }
        MoELayer layer{moe, nullptr, nullptr};
        auto topk_source = moe->get_input_node_shared_ptr(2);
        if (ov::is_type<ov::op::v0::Convert>(topk_source)) {
            topk_source = topk_source->get_input_node_shared_ptr(0);
        }
        layer.router = ov::as_type_ptr<ov::intel_gpu::op::MoERouterFused>(topk_source);
        if (layer.router) {
            layer.router_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(layer.router->get_input_node_shared_ptr(0));
        }
        layers.push_back(layer);
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].moe->get_rt_info()[layer_index_key] = static_cast<int64_t>(i);
    }
    if (m_lookahead_k == 0 || layers.size() < 2) {
        return false;
    }

    bool changed = false;
    for (size_t i = 0; i + 1 < layers.size(); ++i) {
        const auto& cur = layers[i];
        const auto& next = layers[i + 1];
        if (!cur.router_matmul || !next.router_matmul) {
            continue;
        }

        std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned;
        auto weights = clone_weight_subgraph(next.router_matmul->input_value(1), cloned);
        auto logits = std::make_shared<ov::op::v0::MatMul>(cur.router_matmul->input_value(0),
                                                            weights,
                                                            next.router_matmul->get_transpose_a(),
                                                            next.router_matmul->get_transpose_b());
        logits->set_friendly_name(cur.moe->get_friendly_name() + "/lookahead_logits");
        ov::copy_runtime_info(next.router_matmul, logits);

        auto router_config = next.router->get_config();
        if (m_lookahead_k != use_router_top_k) {
            router_config.top_k = std::min(m_lookahead_k, router_config.num_expert);
        }
        ov::OutputVector router_args{logits};
        for (size_t j = 1; j < next.router->get_input_size(); ++j) {
            router_args.push_back(next.router->input_value(j));
        }
        auto hint = std::make_shared<ov::intel_gpu::op::MoERouterFused>(router_args, router_config);
        hint->set_friendly_name(cur.moe->get_friendly_name() + "/lookahead_router");
        ov::copy_runtime_info(next.router, hint);

        auto args = cur.moe->input_values();
        args.push_back(hint->output(1));
        auto moe = std::make_shared<ov::op::internal::MOECompressed>(args, cur.moe->get_config());
        moe->set_friendly_name(cur.moe->get_friendly_name());
        ov::copy_runtime_info(cur.moe, moe);
        moe->get_rt_info()[layer_index_key] = static_cast<int64_t>(i);
        ov::replace_node(cur.moe, moe);
        changed = true;
    }
    return changed;
}

}  // namespace ov::intel_gpu
