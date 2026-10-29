// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstring>
#include <vector>

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;
using ov::intel_gpu::ocl::moe_otd::maybe_transpose_scale_zp;

namespace {

MOECompressed::Config make_transpose_test_config() {
    MOECompressed::Config config{};
    config.hidden_size = 6;
    config.inter_size = 4;
    config.group_size = 2;
    return config;
}

std::vector<uint8_t> make_test_bytes(size_t size) {
    std::vector<uint8_t> bytes(size);
    for (size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    }
    return bytes;
}

std::vector<uint8_t> reference_byte_transpose(const std::vector<uint8_t>& src, size_t oc, size_t group_count, size_t element_bytes) {
    std::vector<uint8_t> dst(src.size());
    for (size_t o = 0; o < oc; ++o) {
        for (size_t g = 0; g < group_count; ++g) {
            const size_t src_offset = (o * group_count + g) * element_bytes;
            const size_t dst_offset = (g * oc + o) * element_bytes;
            std::memcpy(dst.data() + dst_offset, src.data() + src_offset, element_bytes);
        }
    }
    return dst;
}

std::vector<uint8_t> reference_u4_transpose(const std::vector<uint8_t>& src, size_t oc, size_t group_count) {
    std::vector<uint8_t> dst(src.size(), 0);
    for (size_t o = 0; o < oc; ++o) {
        for (size_t g = 0; g < group_count; ++g) {
            const size_t src_idx = o * group_count + g;
            const size_t dst_idx = g * oc + o;
            const uint8_t value = static_cast<uint8_t>((src[src_idx / 2] >> ((src_idx % 2) * 4)) & 0x0F);
            dst[dst_idx / 2] |= static_cast<uint8_t>(value << ((dst_idx % 2) * 4));
        }
    }
    return dst;
}

}  // namespace

TEST(moe_otd_transpose, scale_f16_and_f32_match_reference) {
    const auto config = make_transpose_test_config();
    for (const size_t offset_pos : {3U, 5U}) {
        const bool is_down = offset_pos % 3 == 2;
        const size_t oc = is_down ? config.hidden_size : config.inter_size;
        const size_t ic = is_down ? config.inter_size : config.hidden_size;
        const size_t group_count = ic / config.group_size;
        const data_types scale_types[] = {data_types::f16, data_types::f32};
        const size_t element_bytes[] = {2, 4};
        for (size_t type_idx = 0; type_idx < 2; ++type_idx) {
            auto payload = make_test_bytes(oc * group_count * element_bytes[type_idx]);
            const auto expected = reference_byte_transpose(payload, oc, group_count, element_bytes[type_idx]);
            const layout tensor_layout(ov::Shape{1, 1, 1, oc * group_count}, scale_types[type_idx], format::bfyx);

            maybe_transpose_scale_zp(config, offset_pos, tensor_layout, payload, expected.size());

            EXPECT_EQ(payload, expected) << "offset_pos=" << offset_pos << ", element_bytes=" << element_bytes[type_idx];
        }
    }
}

TEST(moe_otd_transpose, zp_u8_and_u4_match_reference) {
    const auto config = make_transpose_test_config();
    for (const size_t offset_pos : {6U, 8U}) {
        const bool is_down = offset_pos % 3 == 2;
        const size_t oc = is_down ? config.hidden_size : config.inter_size;
        const size_t ic = is_down ? config.inter_size : config.hidden_size;
        const size_t group_count = ic / config.group_size;
        const size_t elem_count = oc * group_count;

        {
            auto payload = make_test_bytes(elem_count);
            const auto expected = reference_byte_transpose(payload, oc, group_count, /*element_bytes=*/1);
            const layout tensor_layout(ov::Shape{1, 1, 1, elem_count}, data_types::u8, format::bfyx);
            maybe_transpose_scale_zp(config, offset_pos, tensor_layout, payload, expected.size());
            EXPECT_EQ(payload, expected) << "offset_pos=" << offset_pos << ", type=u8";
        }

        {
            auto payload = make_test_bytes(elem_count / 2);
            const auto expected = reference_u4_transpose(payload, oc, group_count);
            const layout tensor_layout(ov::Shape{1, 1, 1, elem_count}, data_types::u4, format::bfyx);
            maybe_transpose_scale_zp(config, offset_pos, tensor_layout, payload, expected.size());
            EXPECT_EQ(payload, expected) << "offset_pos=" << offset_pos << ", type=u4";
        }
    }
}

TEST(moe_expert_weight_provider, resident_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_FALSE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 0U);

    // Fully resident: try_acquire_simultaneous returns the expert ids unchanged, including duplicates
    // (no remap, no dedup) - the expert id is already the addressable slot.
    std::vector<uint32_t> experts = {3, 1, 4, 1, 5, 9, 2, 6};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 1, 4, 1, 5, 9, 2, 6}));
}

TEST(moe_expert_weight_provider, offload_construction_state) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    OffloadExpertWeightProvider provider(/*capacity=*/16, config, offsets, "/path/to/weights.bin");

    ASSERT_TRUE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 16U);
    // Resident buffers are bound lazily at first execute, not at construction.
    ASSERT_FALSE(provider.is_bound());
    ASSERT_FALSE(provider.cache().is_initialized());
}

TEST(moe_expert_weight_provider, offload_bind_resident_buffers) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/8, config, {}, "/path/to/weights.bin");
    ASSERT_FALSE(provider.is_bound());

    cldnn::moe_weights resident;  // null device pointers are fine; only the binding flag matters here
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
    ASSERT_TRUE(provider.cache().is_initialized());

    // Idempotent: a second bind keeps the provider bound.
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
}

TEST(moe_expert_weight_provider, resident_try_acquire_simultaneous_is_identity) {
    ResidentExpertWeightProvider provider;
    std::vector<uint32_t> experts = {3, 7, 3, 1, 7};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 7, 3, 1, 7}));
}

TEST(moe_expert_weight_provider, resident_acquire_one_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_EQ(provider.acquire_one(42, get_test_stream()), 42U);
    ASSERT_EQ(provider.acquire_one(0, get_test_stream()), static_cast<size_t>(0));
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_capacity_overflow) {
    // Capacity=2 means at most 2 unique experts can be acquired simultaneously.
    // Requesting 3 unique experts must return nullopt (capacity overflow).
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate the LRU cache so experts 0,1 are hits (no disk I/O needed)
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 3 unique experts (0, 1, 2) exceeds capacity=2 -> nullopt
    std::vector<uint32_t> experts = {0, 1, 2, 0, 1};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_FALSE(lease.has_value());
}

TEST(moe_expert_weight_provider, offload_capacity_overflow_does_not_load_or_mutate_cache) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // The first two experts miss. Rejecting only after acquiring them would attempt file I/O.
    const std::vector<uint32_t> experts = {7, 3, 5, 7};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());

    ASSERT_FALSE(lease.has_value());
    EXPECT_EQ(provider.cache().size(), 0U);
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_within_capacity) {
    // Same setup but with only 2 unique experts -> should succeed
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 2 unique experts (0, 1) with duplicates -> within capacity=2
    std::vector<uint32_t> experts = {0, 1, 0, 1, 0};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    // Slots should be valid (cache returned slots for experts 0 and 1)
    ASSERT_EQ(lease->size(), 5U);
    // Duplicates must map to the same slot
    ASSERT_EQ((*lease)[0], (*lease)[2]);
    ASSERT_EQ((*lease)[0], (*lease)[4]);
    ASSERT_EQ((*lease)[1], (*lease)[3]);
}

TEST(moe_expert_weight_provider, offload_acquire_one_basic) {
    // acquire_one always succeeds (evicts LRU if needed)
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    auto [slot0, hit0] = cache.get_lru_item(5);
    cache.set_filled(slot0);

    // acquire_one should return a valid slot
    auto slot = provider.acquire_one(5, get_test_stream());
    ASSERT_EQ(slot, slot0);
}
