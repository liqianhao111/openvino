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
using ov::intel_gpu::ocl::moe::ExpertSlotCache;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;
using ov::intel_gpu::ocl::moe::SlotState;
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

size_t make_resident(ExpertSlotCache& cache, uint32_t expert) {
    auto acquired = cache.acquire({expert});
    OPENVINO_ASSERT(acquired.has_value());
    const size_t slot = (*acquired)[0].slot;
    cache.mark_ready(slot, (*acquired)[0].generation);
    cache.unpin(slot);
    return slot;
}

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

namespace {

template <typename Pred>
bool spin_until(Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

}  // namespace

TEST(moe_otd_io_engine, submit_returns_before_tasks_finish) {
    MoeIoEngine engine(2);
    std::atomic<bool> release{false};
    std::atomic<size_t> completed{0};
    std::vector<MoeIoEngine::Task> tasks(6, [&](IoWorkerContext&) {
        ASSERT_TRUE(spin_until([&] {
            return release.load();
        }));
        ++completed;
    });
    auto ticket = engine.submit(std::move(tasks));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(completed.load(), 0U);
    EXPECT_FALSE(MoeIoEngine::is_done(ticket));

    release = true;
    engine.wait(ticket);
    EXPECT_EQ(completed.load(), 6U);
    EXPECT_TRUE(MoeIoEngine::is_done(ticket));

    auto empty = engine.submit({});
    EXPECT_TRUE(MoeIoEngine::is_done(empty));
    engine.wait(empty);
}

TEST(moe_otd_io_engine, wait_executes_unstarted_tasks_on_caller) {
    MoeIoEngine engine(1);
    std::atomic<bool> blocker_started{false};
    std::atomic<bool> release{false};
    auto blocker = engine.submit({[&](IoWorkerContext&) {
        blocker_started = true;
        ASSERT_TRUE(spin_until([&] {
            return release.load();
        }));
    }});
    ASSERT_TRUE(spin_until([&] {
        return blocker_started.load();
    }));

    // The only worker is busy, so wait() must run these on the calling thread.
    const auto caller = std::this_thread::get_id();
    std::atomic<size_t> on_caller{0};
    std::vector<MoeIoEngine::Task> tasks(4, [&](IoWorkerContext&) {
        if (std::this_thread::get_id() == caller) {
            ++on_caller;
        }
    });
    engine.wait(engine.submit(std::move(tasks)));
    EXPECT_EQ(on_caller.load(), 4U);

    release = true;
    engine.wait(blocker);
}

TEST(moe_otd_io_engine, urgent_batches_run_before_prefetch) {
    MoeIoEngine engine(1);
    std::atomic<bool> blocker_started{false};
    std::atomic<bool> release{false};
    auto blocker = engine.submit({[&](IoWorkerContext&) {
        blocker_started = true;
        ASSERT_TRUE(spin_until([&] {
            return release.load();
        }));
    }});
    ASSERT_TRUE(spin_until([&] {
        return blocker_started.load();
    }));

    std::mutex order_mutex;
    std::vector<char> order;
    std::atomic<size_t> completed{0};
    auto make_tasks = [&](char tag) {
        return std::vector<MoeIoEngine::Task>(3, [&, tag](IoWorkerContext&) {
            {
                std::lock_guard<std::mutex> lock(order_mutex);
                order.push_back(tag);
            }
            ++completed;
        });
    };
    auto prefetch = engine.submit(make_tasks('p'), ov::intel_gpu::ocl::moe_otd::IoPriority::prefetch);
    auto urgent = engine.submit(make_tasks('u'), ov::intel_gpu::ocl::moe_otd::IoPriority::urgent);

    release = true;
    ASSERT_TRUE(spin_until([&] {
        return completed.load() == 6;
    }));
    EXPECT_EQ(order, (std::vector<char>{'u', 'u', 'u', 'p', 'p', 'p'}));
    engine.wait(blocker);
    engine.wait(prefetch);
    engine.wait(urgent);
}

TEST(moe_otd_io_engine, async_wait_rethrows_task_exception) {
    MoeIoEngine engine(2);
    std::vector<MoeIoEngine::Task> failing;
    for (size_t i = 0; i < 6; ++i) {
        failing.emplace_back([i](IoWorkerContext&) {
            if (i == 2) {
                throw std::runtime_error("io failure");
            }
        });
    }
    auto ticket = engine.submit(std::move(failing), ov::intel_gpu::ocl::moe_otd::IoPriority::prefetch);
    EXPECT_THROW(engine.wait(ticket), std::runtime_error);
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

    // Async request on a 2-slot provider over the same buffers: hits are ready at once, misses after wait().
    OffloadExpertWeightProvider small(/*capacity=*/2, config, offsets, file.path());
    small.bind(resident);
    auto first = small.request({3}, get_test_stream());
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->ready, (std::vector<uint8_t>{0}));
    EXPECT_FALSE(first->all_ready());
    small.wait(*first);
    EXPECT_TRUE(first->all_ready());
    small.release(*first);
    expect_slot_holds_expert(first->slots[0], 3);

    auto mixed = small.request({0, 3, 0}, get_test_stream());
    ASSERT_TRUE(mixed.has_value());
    EXPECT_EQ(mixed->ready, (std::vector<uint8_t>{0, 1, 0}));
    EXPECT_EQ(mixed->slots[1], first->slots[0]);
    EXPECT_EQ(mixed->slots[0], mixed->slots[2]);
    EXPECT_EQ(small.cache().pin_count(mixed->slots[0]), 1U);
    EXPECT_EQ(small.cache().pin_count(mixed->slots[1]), 1U);
    // Both slots are pinned, so a third expert cannot be placed.
    EXPECT_FALSE(small.request({1}, get_test_stream()).has_value());
    small.wait(*mixed);
    EXPECT_EQ(small.cache().state(mixed->slots[0]), SlotState::ready);
    small.release(*mixed);
    EXPECT_EQ(small.cache().pin_count(mixed->slots[0]), 0U);
    expect_slot_holds_expert(mixed->slots[0], 0);
    expect_slot_holds_expert(mixed->slots[1], 3);
}

TEST(moe_expert_slot_cache, hits_are_pinned_before_eviction) {
    ExpertSlotCache cache(2);
    const size_t s0 = make_resident(cache, 10);
    const size_t s1 = make_resident(cache, 11);
    ASSERT_NE(s0, s1);

    // 10 is least recently used, but it is requested together with the miss, so 11 is evicted.
    auto acquired = cache.acquire({12, 10});
    ASSERT_TRUE(acquired.has_value());
    EXPECT_EQ((*acquired)[0].slot, s1);
    EXPECT_EQ((*acquired)[0].state, SlotState::empty);
    EXPECT_EQ((*acquired)[1].slot, s0);
    EXPECT_EQ((*acquired)[1].state, SlotState::ready);
    EXPECT_EQ(cache.state(s1), SlotState::loading);
    EXPECT_FALSE(cache.find(11).has_value());
}

TEST(moe_expert_slot_cache, loading_and_pinned_slots_are_not_evicted) {
    ExpertSlotCache cache(3);
    const size_t ready = make_resident(cache, 1);
    auto loading = cache.acquire({2});
    ASSERT_TRUE(loading.has_value());
    cache.unpin((*loading)[0].slot);  // unpinned but still LOADING
    auto pinned = cache.acquire({3});
    ASSERT_TRUE(pinned.has_value());
    cache.mark_ready((*pinned)[0].slot, (*pinned)[0].generation);  // READY but pinned

    // Only the slot of expert 1 is evictable: one miss fits, two do not and leave the cache unchanged.
    EXPECT_FALSE(cache.acquire({4, 5}).has_value());
    EXPECT_EQ(cache.find(1).value_or(SIZE_MAX), ready);
    auto one = cache.acquire({4});
    ASSERT_TRUE(one.has_value());
    EXPECT_EQ((*one)[0].slot, ready);
    EXPECT_FALSE(cache.find(1).has_value());
}

TEST(moe_expert_slot_cache, evicts_empty_then_least_recently_used) {
    ExpertSlotCache cache(3);
    const size_t a = make_resident(cache, 1);
    const size_t b = make_resident(cache, 2);
    const size_t c = make_resident(cache, 3);
    EXPECT_EQ((std::vector<size_t>{a, b, c}), (std::vector<size_t>{0, 1, 2}));

    make_resident(cache, 1);  // touch: LRU order is now 2, 3, 1
    auto failed = cache.acquire({4});
    ASSERT_TRUE(failed.has_value());
    EXPECT_EQ((*failed)[0].slot, b);
    cache.mark_failed(b, (*failed)[0].generation);
    cache.unpin(b);
    EXPECT_FALSE(cache.find(4).has_value());
    EXPECT_EQ(cache.state(b), SlotState::empty);

    // The emptied slot is reused before evicting the LRU expert 3.
    EXPECT_EQ(make_resident(cache, 5), b);
    EXPECT_EQ(make_resident(cache, 6), c);
    EXPECT_EQ(cache.size(), 3U);
}

TEST(moe_expert_slot_cache, stale_generation_does_not_change_reassigned_slot) {
    ExpertSlotCache cache(1);
    auto first = cache.acquire({1}, /*prefetch=*/true);
    ASSERT_TRUE(first.has_value());
    const auto old = (*first)[0];
    cache.mark_ready(old.slot, old.generation);
    cache.unpin(old.slot);

    auto second = cache.acquire({2});
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ((*second)[0].slot, old.slot);
    EXPECT_TRUE((*second)[0].evicted_unused_prefetch);
    // A late completion or failure of the old load must not touch the new assignment.
    cache.mark_ready(old.slot, old.generation);
    EXPECT_EQ(cache.state(old.slot), SlotState::loading);
    cache.mark_failed(old.slot, old.generation);
    EXPECT_EQ(cache.find(2).value_or(SIZE_MAX), old.slot);
    cache.mark_ready(old.slot, (*second)[0].generation);
    EXPECT_EQ(cache.state(old.slot), SlotState::ready);
}

TEST(moe_expert_slot_cache, prefetch_hit_is_reported_once) {
    ExpertSlotCache cache(2);
    auto prefetched = cache.acquire({7}, /*prefetch=*/true);
    ASSERT_TRUE(prefetched.has_value());
    cache.unpin((*prefetched)[0].slot);
    auto first = cache.acquire({7});
    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE((*first)[0].prefetch_hit);
    EXPECT_EQ((*first)[0].state, SlotState::loading);
    cache.unpin((*first)[0].slot);
    auto again = cache.acquire({7});
    ASSERT_TRUE(again.has_value());
    EXPECT_FALSE((*again)[0].prefetch_hit);
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
    ASSERT_EQ(provider.cache().size(), 0U);
}

TEST(moe_expert_weight_provider, offload_bind_resident_buffers) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/8, config, {}, "/path/to/weights.bin");
    ASSERT_FALSE(provider.is_bound());

    cldnn::moe_weights resident;  // null device pointers are fine; only the binding flag matters here
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());

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

    // Pre-populate the cache so experts 0,1 are hits (no disk I/O needed)
    make_resident(provider.cache(), 0);
    make_resident(provider.cache(), 1);

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
    make_resident(provider.cache(), 0);
    make_resident(provider.cache(), 1);

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
    const size_t slot0 = make_resident(provider.cache(), 5);

    // acquire_one should return a valid slot
    auto slot = provider.acquire_one(5, get_test_stream());
    ASSERT_EQ(slot, slot0);
}
