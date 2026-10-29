// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/moe_otd_io.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;
using ov::intel_gpu::ocl::moe_otd::IoWorkerContext;
using ov::intel_gpu::ocl::moe_otd::maybe_transpose_scale_zp;
using ov::intel_gpu::ocl::moe_otd::MoeIoEngine;

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

class TempWeightsFile {
public:
    explicit TempWeightsFile(const std::vector<uint8_t>& bytes) {
        _path = std::filesystem::temp_directory_path() /
                ("moe_otd_io_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
        std::ofstream out(_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    ~TempWeightsFile() {
        std::error_code ec;
        std::filesystem::remove(_path, ec);
    }
    const std::filesystem::path& path() const {
        return _path;
    }

private:
    std::filesystem::path _path;
};

}  // namespace

TEST(moe_otd_io_engine, runs_tasks_concurrently) {
    MoeIoEngine engine(4);
    std::atomic<size_t> completed{0};
    std::atomic<size_t> running{0};
    std::atomic<size_t> peak{0};
    std::vector<MoeIoEngine::Task> tasks;
    for (size_t i = 0; i < 32; ++i) {
        tasks.emplace_back([&](IoWorkerContext&) {
            const size_t now = ++running;
            size_t prev = peak.load();
            while (prev < now && !peak.compare_exchange_weak(prev, now)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            --running;
            ++completed;
        });
    }
    engine.run(std::move(tasks));
    EXPECT_EQ(completed.load(), 32U);
    EXPECT_GE(peak.load(), 2U);
}

TEST(moe_otd_io_engine, propagates_task_exception_and_stays_usable) {
    MoeIoEngine engine(2);
    std::vector<MoeIoEngine::Task> failing;
    for (size_t i = 0; i < 8; ++i) {
        failing.emplace_back([i](IoWorkerContext&) {
            if (i == 3) {
                throw std::runtime_error("io failure");
            }
        });
    }
    EXPECT_THROW(engine.run(std::move(failing)), std::runtime_error);

    std::atomic<size_t> completed{0};
    std::vector<MoeIoEngine::Task> tasks(5, [&](IoWorkerContext&) {
        ++completed;
    });
    engine.run(std::move(tasks));
    EXPECT_EQ(completed.load(), 5U);
}

TEST(moe_otd_io_engine, worker_reads_match_file) {
    const auto bytes = make_test_bytes(64 * 1024 + 123);
    TempWeightsFile file(bytes);
    std::vector<uint8_t> out(bytes.size());

    MoeIoEngine engine(3);
    constexpr size_t chunk = 4096;
    std::vector<MoeIoEngine::Task> tasks;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk) {
        const size_t size = std::min(chunk, bytes.size() - offset);
        tasks.emplace_back([&, offset, size](IoWorkerContext& context) {
            context.read(file.path(), reinterpret_cast<char*>(out.data() + offset), size, offset);
        });
    }
    engine.run(std::move(tasks));
    EXPECT_EQ(out, bytes);

    std::vector<MoeIoEngine::Task> past_eof;
    past_eof.emplace_back([&](IoWorkerContext& context) {
        context.read(file.path(), reinterpret_cast<char*>(out.data()), 16, bytes.size());
    });
    EXPECT_ANY_THROW(engine.run(std::move(past_eof)));
}

TEST(moe_expert_weight_provider, offload_parallel_load_matches_reference) {
    auto& engine = get_test_engine();
    if (!engine.supports_allocation(allocation_type::usm_host)) {
        GTEST_SKIP() << "usm_host is not supported";
    }

    MOECompressed::Config config{};
    config.num_expert = 4;
    config.hidden_size = 1024;
    config.inter_size = 512;
    config.group_size = 64;
    const size_t num_expert = config.num_expert;
    const size_t gate_up_groups = config.hidden_size / config.group_size;
    const size_t down_groups = config.inter_size / config.group_size;

    // Offset order: gate_w, up_w, down_w, gate_s, up_s, down_s, gate_z, up_z, down_z.
    const std::vector<layout> layouts = {
        layout(ov::PartialShape{4, 512, 1024}, data_types::u4, format::bfyx),
        layout(ov::PartialShape{4, 512, 1024}, data_types::u4, format::bfyx),
        layout(ov::PartialShape{4, 1024, 512}, data_types::u4, format::bfyx),
        layout(ov::PartialShape{4, 512, static_cast<int64_t>(gate_up_groups)}, data_types::f16, format::bfyx),
        layout(ov::PartialShape{4, 512, static_cast<int64_t>(gate_up_groups)}, data_types::f16, format::bfyx),
        layout(ov::PartialShape{4, 1024, static_cast<int64_t>(down_groups)}, data_types::f16, format::bfyx),
        layout(ov::PartialShape{4, 512, static_cast<int64_t>(gate_up_groups)}, data_types::u4, format::bfyx),
        layout(ov::PartialShape{4, 512, static_cast<int64_t>(gate_up_groups)}, data_types::u4, format::bfyx),
        layout(ov::PartialShape{4, 1024, static_cast<int64_t>(down_groups)}, data_types::u4, format::bfyx),
    };

    std::vector<size_t> offsets;
    std::vector<uint8_t> file_bytes;
    for (const auto& l : layouts) {
        offsets.push_back(file_bytes.size());
        const auto region = make_test_bytes(l.bytes_count());
        file_bytes.insert(file_bytes.end(), region.begin(), region.end());
        file_bytes.push_back(0x5A);  // keep regions unaligned to each other
    }
    TempWeightsFile file(file_bytes);

    std::vector<memory::ptr> mems;
    for (const auto& l : layouts) {
        mems.push_back(engine.allocate_memory(l, allocation_type::usm_host, false));
    }
    moe_weights resident;
    resident.gate_w = mems[0];
    resident.up_w = mems[1];
    resident.down_w = mems[2];
    resident.gate_s = mems[3];
    resident.up_s = mems[4];
    resident.down_s = mems[5];
    resident.gate_z = mems[6];
    resident.up_z = mems[7];
    resident.down_z = mems[8];

    OffloadExpertWeightProvider provider(/*capacity=*/num_expert, config, offsets, file.path());
    provider.bind(resident);

    auto expect_slot_holds_expert = [&](size_t slot, uint32_t expert) {
        for (size_t pos = 0; pos < layouts.size(); ++pos) {
            const size_t per_expert = layouts[pos].bytes_count() / num_expert;
            const auto src_begin = file_bytes.begin() + static_cast<std::ptrdiff_t>(offsets[pos] + expert * per_expert);
            std::vector<uint8_t> expected(src_begin, src_begin + static_cast<std::ptrdiff_t>(per_expert));
            if (pos >= 3) {
                const bool is_down = pos % 3 == 2;
                const size_t oc = is_down ? config.hidden_size : config.inter_size;
                const size_t groups = is_down ? down_groups : gate_up_groups;
                expected = pos < 6 ? reference_byte_transpose(expected, oc, groups, 2) : reference_u4_transpose(expected, oc, groups);
            }
            mem_lock<uint8_t, mem_lock_type::read> lock(mems[pos], get_test_stream());
            std::vector<uint8_t> actual(lock.data() + slot * per_expert, lock.data() + (slot + 1) * per_expert);
            EXPECT_EQ(actual, expected) << "tensor=" << pos << ", slot=" << slot << ", expert=" << expert;
        }
    };

    const std::vector<uint32_t> experts = {3, 1, 3, 2};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ((*lease)[0], (*lease)[2]);
    for (size_t i = 0; i < experts.size(); ++i) {
        expect_slot_holds_expert((*lease)[i], experts[i]);
    }

    const size_t slot0 = provider.acquire_one(0, get_test_stream());
    expect_slot_holds_expert(slot0, 0);

    // All four experts are now resident; a repeat request must hit without reloading.
    auto repeat = provider.try_acquire_simultaneous({0, 1, 2, 3}, get_test_stream());
    ASSERT_TRUE(repeat.has_value());
    EXPECT_EQ((*repeat)[1], (*lease)[1]);
    EXPECT_EQ((*repeat)[0], slot0);
}

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
