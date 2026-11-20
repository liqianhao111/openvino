// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_io.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <string_view>

#include "openvino/core/except.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

namespace {

// Covers both 512-byte and 4 KiB logical block sizes.
constexpr size_t direct_io_alignment = 4096;

// GPU_MOE_OTD_DIRECT_IO=1 bypasses the page cache so expert loads hit the SSD.
bool direct_io_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GPU_MOE_OTD_DIRECT_IO");
        return env != nullptr && std::string_view(env) == "1";
    }();
    return enabled;
}

}  // namespace

IoWorkerContext::~IoWorkerContext() {
    for (const auto& entry : _handles) {
        ov::util::close_file_handle(entry.second);
    }
    for (const auto& file : _direct_files) {
        ov::util::close_file(file.handle);
    }
}

void IoWorkerContext::read(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset) {
    if (direct_io_enabled()) {
        read_direct(path, dst, size, file_offset);
    } else {
        read_buffered(path, dst, size, file_offset);
    }
}

const char* IoWorkerContext::read_view(const std::filesystem::path& path, size_t size, size_t file_offset) {
    if (direct_io_enabled()) {
        auto& file = direct_file(path);
        if (file_offset + size <= (file.size & ~(direct_io_alignment - 1))) {
            return read_direct_span(file, size, file_offset);
        }
    }
    if (_view_buffer.size() < size) {
        _view_buffer.resize(size);
    }
    auto* dst = reinterpret_cast<char*>(_view_buffer.data());
    read(path, dst, size, file_offset);
    return dst;
}

IoWorkerContext::DirectFile& IoWorkerContext::direct_file(const std::filesystem::path& path) {
    auto file = std::find_if(_direct_files.begin(), _direct_files.end(), [&](const DirectFile& f) {
        return f.path == path;
    });
    if (file != _direct_files.end()) {
        return *file;
    }
    const auto handle = ov::util::open_file(path, ov::util::FileMode::READ | ov::util::FileMode::DIRECT);
    OPENVINO_ASSERT(handle != ov::invalid_handle, "Failed to open OTD weight file with direct IO: ", ov::util::path_to_string(path));
    _direct_files.push_back({path, handle, static_cast<size_t>(std::filesystem::file_size(path))});
    return _direct_files.back();
}

// Reads the aligned span covering [file_offset, file_offset + size) into the bounce buffer; the span must not
// extend past the aligned file size.
const char* IoWorkerContext::read_direct_span(DirectFile& file, size_t size, size_t file_offset) {
    constexpr size_t align = direct_io_alignment;
    const size_t begin = file_offset & ~(align - 1);
    const size_t end = (file_offset + size + align - 1) & ~(align - 1);
    const size_t span = end - begin;
    if (_direct_buffer.size() < span + align) {
        _direct_buffer.resize(span + align);
    }
    const auto base = reinterpret_cast<uintptr_t>(_direct_buffer.data());
    auto* aligned = reinterpret_cast<char*>((base + align - 1) & ~static_cast<uintptr_t>(align - 1));
    if (!ov::util::positional_read(file.handle, aligned, span, begin)) {
        OPENVINO_THROW("Failed to read enough bytes from OTD weight file with direct IO");
    }
    return aligned + (file_offset - begin);
}

void IoWorkerContext::read_direct(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset) {
    auto& file = direct_file(path);
    constexpr size_t align = direct_io_alignment;
    if (reinterpret_cast<uintptr_t>(dst) % align == 0 && file_offset % align == 0 && size % align == 0 &&
        file_offset + size <= file.size) {
        if (!ov::util::positional_read(file.handle, dst, size, file_offset)) {
            OPENVINO_THROW("Failed to read enough bytes from OTD weight file with direct IO");
        }
        return;
    }

    // Direct IO cannot read the unaligned file tail; it falls back to a buffered read.
    const size_t direct_end = std::min(file_offset + size, file.size & ~(align - 1));
    if (direct_end > file_offset) {
        const size_t copied = direct_end - file_offset;
        std::memcpy(dst, read_direct_span(file, copied, file_offset), copied);
        dst += copied;
        file_offset += copied;
        size -= copied;
    }
    if (size > 0) {
        read_buffered(path, dst, size, file_offset);
    }
}

void IoWorkerContext::read_buffered(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset) {
    ov::FileHandle handle = ov::invalid_handle;
    for (const auto& entry : _handles) {
        if (entry.first == path) {
            handle = entry.second;
            break;
        }
    }
    if (handle == ov::invalid_handle) {
        handle = ov::util::open_file_for_read(path);
        OPENVINO_ASSERT(handle != ov::invalid_handle, "Failed to open OTD weight file: ", ov::util::path_to_string(path));
        _handles.emplace_back(path, handle);
    }
    if (!ov::util::positional_read(handle, dst, size, file_offset)) {
        OPENVINO_THROW("Failed to read enough bytes from OTD weight file");
    }
}

struct MoeIoEngine::Batch {
    std::vector<Task> tasks;
    size_t next = 0;  // guarded by the engine mutex
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::condition_variable done_cv;
    size_t done = 0;
    std::exception_ptr error;
};

MoeIoEngine::MoeIoEngine(size_t num_threads) {
    OPENVINO_ASSERT(num_threads > 0, "MoeIoEngine requires at least one worker thread");
    _workers.reserve(num_threads);
    try {
        for (size_t i = 0; i < num_threads; ++i) {
            _workers.emplace_back([this] {
                worker_loop();
            });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& worker : _workers) {
            worker.join();
        }
        throw;
    }
}

MoeIoEngine::~MoeIoEngine() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stop = true;
    }
    _cv.notify_all();
    for (auto& worker : _workers) {
        worker.join();
    }
}

size_t MoeIoEngine::configured_thread_count() {
    constexpr size_t max_threads = 64;
    if (const char* env = std::getenv("GPU_MOE_OTD_IO_THREADS"); env && *env) {
        char* end = nullptr;
        const unsigned long value = std::strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            return std::min<size_t>(static_cast<size_t>(value), max_threads);
        }
    }
    return std::clamp<size_t>(std::thread::hardware_concurrency() / 2, 1, 8);
}

std::shared_ptr<MoeIoEngine> MoeIoEngine::shared() {
    static std::mutex mutex;
    static std::weak_ptr<MoeIoEngine> instance;

    std::lock_guard<std::mutex> lock(mutex);
    if (auto engine = instance.lock()) {
        return engine;
    }
    const size_t threads = configured_thread_count();
    if (threads == 0) {
        return nullptr;
    }
    auto engine = std::make_shared<MoeIoEngine>(threads);
    instance = engine;
    return engine;
}

MoeIoEngine::Ticket MoeIoEngine::submit(std::vector<Task> tasks, IoPriority priority) {
    auto batch = std::make_shared<Batch>();
    batch->tasks = std::move(tasks);
    if (batch->tasks.empty()) {
        return batch;
    }

    const size_t wake_count = std::min(batch->tasks.size(), _workers.size());
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queues[static_cast<size_t>(priority)].push_back(batch);
    }
    if (wake_count == _workers.size()) {
        _cv.notify_all();
    } else {
        for (size_t i = 0; i < wake_count; ++i) {
            _cv.notify_one();
        }
    }
    return batch;
}

void MoeIoEngine::wait(const Ticket& ticket) {
    OPENVINO_ASSERT(ticket != nullptr, "MoeIoEngine::wait called with an empty ticket");
    Batch& batch = *ticket;

    std::unique_ptr<IoWorkerContext> context;
    for (;;) {
        size_t index = 0;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (batch.next == batch.tasks.size()) {
                // Workers drop the exhausted batch from the queue when they reach it.
                if (context) {
                    _caller_contexts.push_back(std::move(context));
                }
                break;
            }
            if (!context) {
                if (_caller_contexts.empty()) {
                    context = std::make_unique<IoWorkerContext>();
                } else {
                    context = std::move(_caller_contexts.back());
                    _caller_contexts.pop_back();
                }
            }
            index = batch.next++;
        }
        execute(batch, index, *context);
    }

    std::unique_lock<std::mutex> lock(batch.mutex);
    batch.done_cv.wait(lock, [&] {
        return batch.done == batch.tasks.size();
    });
    if (batch.error) {
        std::rethrow_exception(batch.error);
    }
}

bool MoeIoEngine::is_done(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(ticket->mutex);
    return ticket->done == ticket->tasks.size();
}

void MoeIoEngine::worker_loop() {
    IoWorkerContext context;
    for (;;) {
        std::shared_ptr<Batch> batch;
        size_t index = 0;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            for (;;) {
                _cv.wait(lock, [this] {
                    return _stop || !_queues[0].empty() || !_queues[1].empty();
                });
                auto& queue = !_queues[0].empty() ? _queues[0] : _queues[1];
                if (queue.empty()) {
                    return;
                }
                batch = queue.front();
                if (batch->next < batch->tasks.size()) {
                    index = batch->next++;
                    if (batch->next == batch->tasks.size()) {
                        queue.pop_front();
                    }
                    break;
                }
                queue.pop_front();
            }
        }
        execute(*batch, index, context);
    }
}

void MoeIoEngine::execute(Batch& batch, size_t index, IoWorkerContext& context) {
    std::exception_ptr error;
    if (!batch.failed.load(std::memory_order_relaxed)) {
        try {
            batch.tasks[index](context);
        } catch (...) {
            error = std::current_exception();
            batch.failed.store(true, std::memory_order_relaxed);
        }
    }

    bool finished = false;
    {
        std::lock_guard<std::mutex> lock(batch.mutex);
        if (error && !batch.error) {
            batch.error = error;
        }
        finished = ++batch.done == batch.tasks.size();
    }
    if (finished) {
        batch.done_cv.notify_all();
    }
}

}  // namespace ov::intel_gpu::ocl::moe_otd
