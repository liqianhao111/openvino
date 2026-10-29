// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_slot_cache.hpp"

#include <algorithm>
#include <tuple>

#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl::moe {

ExpertSlotCache::ExpertSlotCache(size_t capacity) : _slots(capacity) {}

std::optional<std::vector<ExpertSlotCache::Acquired>> ExpertSlotCache::acquire(const std::vector<uint32_t>& experts) {
    std::lock_guard<std::mutex> lock(_mutex);

    std::vector<bool> is_hit_slot(_slots.size(), false);
    size_t miss_count = 0;
    for (const uint32_t expert : experts) {
        auto it = _expert_to_slot.find(expert);
        if (it != _expert_to_slot.end()) {
            is_hit_slot[it->second] = true;
        } else {
            ++miss_count;
        }
    }

    std::vector<size_t> victims;
    for (size_t s = 0; s < _slots.size(); ++s) {
        if (!is_hit_slot[s] && _slots[s].pins == 0 && _slots[s].state != SlotState::loading) {
            victims.push_back(s);
        }
    }
    if (victims.size() < miss_count) {
        return std::nullopt;
    }
    auto victim_order = [this](size_t a, size_t b) {
        const auto& sa = _slots[a];
        const auto& sb = _slots[b];
        return std::make_tuple(sa.state == SlotState::ready, sa.last_use, a) < std::make_tuple(sb.state == SlotState::ready, sb.last_use, b);
    };
    std::partial_sort(victims.begin(), victims.begin() + static_cast<std::ptrdiff_t>(miss_count), victims.end(), victim_order);

    std::vector<Acquired> result;
    result.reserve(experts.size());
    size_t next_victim = 0;
    for (const uint32_t expert : experts) {
        auto it = _expert_to_slot.find(expert);
        if (it != _expert_to_slot.end()) {
            auto& slot = _slots[it->second];
            slot.last_use = ++_clock;
            ++slot.pins;
            result.push_back({it->second, slot.state});
            continue;
        }
        const size_t s = victims[next_victim++];
        auto& slot = _slots[s];
        if (slot.expert != no_expert) {
            _expert_to_slot.erase(slot.expert);
        }
        slot.state = SlotState::loading;
        slot.expert = expert;
        slot.pins = 1;
        slot.last_use = ++_clock;
        _expert_to_slot.emplace(expert, s);
        result.push_back({s, SlotState::empty});
    }
    return result;
}

void ExpertSlotCache::unpin(size_t slot) {
    std::lock_guard<std::mutex> lock(_mutex);
    OPENVINO_ASSERT(_slots.at(slot).pins > 0, "ExpertSlotCache: unpin of an unpinned slot");
    --_slots[slot].pins;
}

void ExpertSlotCache::mark_ready(size_t slot) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& s = _slots.at(slot);
    if (s.state == SlotState::loading) {
        s.state = SlotState::ready;
    }
}

void ExpertSlotCache::mark_failed(size_t slot) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& s = _slots.at(slot);
    if (s.state != SlotState::loading) {
        return;
    }
    _expert_to_slot.erase(s.expert);
    s.state = SlotState::empty;
    s.expert = no_expert;
}

size_t ExpertSlotCache::size() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _expert_to_slot.size();
}

SlotState ExpertSlotCache::state(size_t slot) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _slots.at(slot).state;
}

size_t ExpertSlotCache::pin_count(size_t slot) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _slots.at(slot).pins;
}

std::optional<size_t> ExpertSlotCache::find(uint32_t expert) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _expert_to_slot.find(expert);
    if (it == _expert_to_slot.end()) {
        return std::nullopt;
    }
    return it->second;
}

}  // namespace ov::intel_gpu::ocl::moe
