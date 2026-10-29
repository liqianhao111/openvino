// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ov::intel_gpu::ocl::moe {

enum class SlotState : uint8_t { empty, loading, ready };

// Per-layer map from experts to resident weight slots with an explicit load state.
// Only READY or EMPTY slots without pins are evicted, least recently used first, so a slot that is
// being written by an IO worker or read by an in-flight request is never reassigned.
class ExpertSlotCache {
public:
    explicit ExpertSlotCache(size_t capacity);

    struct Acquired {
        size_t slot;
        // State before this call; EMPTY means the slot was just assigned and the caller must load it.
        SlotState state;
    };

    // Pins one slot per expert (experts must be unique). Hits are pinned before any eviction.
    // Newly assigned slots move to LOADING. Returns nullopt without changing anything when the
    // evictable slots cannot hold all misses.
    std::optional<std::vector<Acquired>> acquire(const std::vector<uint32_t>& experts);

    void unpin(size_t slot);
    void mark_ready(size_t slot);
    // Drops the expert mapping of a LOADING slot whose load did not complete.
    void mark_failed(size_t slot);

    size_t capacity() const {
        return _slots.size();
    }
    // Number of slots that hold or are loading an expert.
    size_t size() const;
    SlotState state(size_t slot) const;
    size_t pin_count(size_t slot) const;
    std::optional<size_t> find(uint32_t expert) const;

private:
    static constexpr uint32_t no_expert = UINT32_MAX;

    struct Slot {
        SlotState state = SlotState::empty;
        uint32_t expert = no_expert;
        uint32_t pins = 0;
        uint64_t last_use = 0;
    };

    std::vector<Slot> _slots;
    std::unordered_map<uint32_t, size_t> _expert_to_slot;
    uint64_t _clock = 0;
    mutable std::mutex _mutex;
};

}  // namespace ov::intel_gpu::ocl::moe
