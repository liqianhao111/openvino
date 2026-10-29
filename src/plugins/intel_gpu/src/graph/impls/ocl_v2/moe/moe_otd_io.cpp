// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_io.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>

#include "openvino/core/except.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

IoWorkerContext::~IoWorkerContext() {
    for (const auto& entry : _handles) {
        ov::util::close_file_handle(entry.second);
    }
}

void IoWorkerContext::read(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset) {
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
    return std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 1, 4);
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

void MoeIoEngine::run(std::vector<Task> tasks) {
    if (tasks.empty()) {
        return;
    }

    auto batch = std::make_shared<Batch>();
    batch->tasks = std::move(tasks);
    // The calling thread executes tasks too, so wake at most one worker per remaining task.
    const size_t wake_count = std::min(batch->tasks.size() - 1, _workers.size());
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push_back(batch);
    }
    if (wake_count == _workers.size()) {
        _cv.notify_all();
    } else {
        for (size_t i = 0; i < wake_count; ++i) {
            _cv.notify_one();
        }
    }

    std::unique_ptr<IoWorkerContext> context;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_caller_contexts.empty()) {
            context = std::make_unique<IoWorkerContext>();
        } else {
            context = std::move(_caller_contexts.back());
            _caller_contexts.pop_back();
        }
    }
    for (;;) {
        size_t index = 0;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (batch->next == batch->tasks.size()) {
                // Workers drop the exhausted batch from the queue when they reach it.
                _caller_contexts.push_back(std::move(context));
                break;
            }
            index = batch->next++;
        }
        execute(*batch, index, *context);
    }

    std::unique_lock<std::mutex> lock(batch->mutex);
    batch->done_cv.wait(lock, [&] {
        return batch->done == batch->tasks.size();
    });
    if (batch->error) {
        std::rethrow_exception(batch->error);
    }
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
                    return _stop || !_queue.empty();
                });
                if (_queue.empty()) {
                    return;
                }
                batch = _queue.front();
                if (batch->next < batch->tasks.size()) {
                    index = batch->next++;
                    if (batch->next == batch->tasks.size()) {
                        _queue.pop_front();
                    }
                    break;
                }
                _queue.pop_front();
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
