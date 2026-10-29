// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "openvino/util/file_util.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

// Per-worker resources handed to every IO task. File handles are per worker because
// ov::util::positional_read is not thread-safe for a shared handle on all platforms.
class IoWorkerContext {
public:
    IoWorkerContext() = default;
    ~IoWorkerContext();

    IoWorkerContext(const IoWorkerContext&) = delete;
    IoWorkerContext& operator=(const IoWorkerContext&) = delete;

    void read(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset);

    std::vector<uint8_t>& staging() {
        return _staging;
    }

private:
    std::vector<std::pair<std::filesystem::path, ov::FileHandle>> _handles;
    std::vector<uint8_t> _staging;
};

enum class IoPriority : size_t { urgent = 0, prefetch = 1 };

// Fixed-size thread pool that executes batches of independent file-read tasks for MoE OTD.
class MoeIoEngine {
public:
    using Task = std::function<void(IoWorkerContext&)>;
    struct Batch;
    using Ticket = std::shared_ptr<Batch>;

    explicit MoeIoEngine(size_t num_threads);
    ~MoeIoEngine();

    MoeIoEngine(const MoeIoEngine&) = delete;
    MoeIoEngine& operator=(const MoeIoEngine&) = delete;

    // Process-wide engine shared by all OTD providers; released with the last user.
    // Returns nullptr when GPU_MOE_OTD_IO_THREADS=0 (legacy serial loading).
    static std::shared_ptr<MoeIoEngine> shared();

    // GPU_MOE_OTD_IO_THREADS if set, otherwise min(4, hardware_concurrency / 4).
    // On PTL B390, 4 workers matched or beat 8 in both hot- and cold-cache runs.
    static size_t configured_thread_count();

    size_t num_threads() const {
        return _workers.size();
    }

    // Queues tasks and returns immediately. Workers drain all urgent batches before prefetch batches.
    Ticket submit(std::vector<Task> tasks, IoPriority priority = IoPriority::urgent);

    // Runs the batch's unstarted tasks on the calling thread, then blocks until every task has finished.
    // Rethrows the first task exception; unstarted tasks of a failed batch are skipped.
    void wait(const Ticket& ticket);

    static bool is_done(const Ticket& ticket);

    void run(std::vector<Task> tasks) {
        wait(submit(std::move(tasks)));
    }

private:
    void worker_loop();
    static void execute(Batch& batch, size_t index, IoWorkerContext& context);

    std::vector<std::thread> _workers;
    std::deque<Ticket> _queues[2];
    // File handles for threads calling run(), reused across calls and closed with the engine.
    std::vector<std::unique_ptr<IoWorkerContext>> _caller_contexts;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _stop = false;
};

}  // namespace ov::intel_gpu::ocl::moe_otd
