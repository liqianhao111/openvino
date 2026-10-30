// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// MoE OTD lookahead: gives every GEMM3 MOECompressed layer L a trailing input with the experts that
// layer L+1's router would pick for layer L's router input (MoERouterFused(x_L * W_{L+1}), top `k`).
// The runtime prefetches them into layer L+1 slots. Every MoE layer gets rt_info "moe_layer_index".
class MoELookaheadRouter : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MoELookaheadRouter");
    explicit MoELookaheadRouter(size_t lookahead_k) : m_lookahead_k(lookahead_k) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    static constexpr const char* layer_index_key = "moe_layer_index";
    static constexpr size_t use_router_top_k = SIZE_MAX;

    // GPU_MOE_OTD_LOOKAHEAD_K: unset = router top_k, 0 = no hint inputs (layer indices are still stamped).
    static size_t configured_lookahead_k();

private:
    size_t m_lookahead_k;
};

}  // namespace ov::intel_gpu
