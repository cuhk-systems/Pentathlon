#ifndef COMPUTE_RDMA_HPP
#define COMPUTE_RDMA_HPP

#include <threads.h>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <tuple>
#include <utility>

#include "common/handshake.h"

#ifndef PENTATHLON_USE_CXL
#include <infiniband/verbs.h>
#include "common/rdma.h"
#include "rdma.h"
#endif

const size_t CACHE_LINE_SIZE = 1024 * 64;  // 64 KiB
const size_t MAX_BATCH_SIZE = 16;
const size_t MAX_RDMA_THREADS = 32;

#ifdef PENTATHLON_USE_CXL
struct CxlBatchEntry {
    void* from = nullptr;
    void* to = nullptr;
    uint32_t size = 0;
    bool is_write = false;
};

struct ThreadBatch {
    std::array<CxlBatchEntry, MAX_BATCH_SIZE> entries;
    size_t count = 0;
    size_t read_count = 0;
    size_t write_count = 0;

    static ThreadBatch& get_instance() {
        static thread_local auto batch = ThreadBatch();
        return batch;
    }

    CxlBatchEntry& get_next() {
        assert(count < MAX_BATCH_SIZE);
        auto& entry = entries[count++];
        entry = {};
        return entry;
    }

    void reset() {
        count = 0;
        read_count = 0;
        write_count = 0;
    }
};
#else
// Thread-local buffer for RDMA operations
struct ThreadBuffer {
    rdma_client* client;
    void* buf;
    ibv_mr* buf_mr;

    ThreadBuffer(rdma_client* client, void* buf, ibv_mr* buf_mr)
        : client(client), buf(buf), buf_mr(buf_mr) {}

    static ThreadBuffer& get_instance(rdma_client* client, void* buf, ibv_mr* buf_mr) {
        static thread_local auto thread_buf = ThreadBuffer(client, buf, buf_mr);
        return thread_buf;
    }

    ~ThreadBuffer() {
        if (buf_mr) ibv_dereg_mr(buf_mr);
        if (buf) free(buf);
    }
};

struct ThreadBatch {
    std::pair<ibv_send_wr, ibv_sge> wrs[MAX_BATCH_SIZE];
    size_t count = 0;
    std::tuple<void*, void*, uint32_t> copy_after_read[MAX_BATCH_SIZE];
    size_t copy_after_read_count = 0;

    size_t read_count = 0;
    size_t write_count = 0;

    static ThreadBatch& get_instance() {
        static thread_local auto batch = ThreadBatch();
        return batch;
    }

    std::pair<ibv_send_wr, ibv_sge>& get_next() {
        assert(count < MAX_BATCH_SIZE);
        auto& result = wrs[count];
        result.first = {};
        result.second = {};
        result.first.sg_list = &result.second;
        result.first.num_sge = 1;
        if (count > 0) wrs[count - 1].first.next = &result.first;
        count++;
        return result;
    }

    void push_copy_after_read(void* from, void* to, uint32_t size) {
        assert(copy_after_read_count < MAX_BATCH_SIZE);
        auto& x = copy_after_read[copy_after_read_count++];
        std::get<0>(x) = from;
        std::get<1>(x) = to;
        std::get<2>(x) = size;
    }

    void reset() {
        count = 0;
        copy_after_read_count = 0;
        read_count = 0;
        write_count = 0;
    }
};
#endif

// C++ wrapper of C `struct rdma_client` with thread safety.
class RDMAClient {
#ifdef PENTATHLON_USE_CXL
    memory_info remote_mem = {};
    void* cxl_pool = nullptr;
    size_t cxl_pool_size = 0;
    int cxl_node = 1;

    void init_cxl_pool();
    void validate_cxl_range(void* ptr, uint32_t size) const;
#else
    // rdma_client* inner;

    ThreadBuffer& get_thread_buffer() {
        int id = get_thread_id();
        ensure_thread_resources(id);
        return ThreadBuffer::get_instance(thread_clients[id], thread_bufs[id], thread_mrs[id]);
    }

    void ensure_thread_resources(int id);
    void init_thread_resources(int id);

    sockaddr_storage addr_;
    memory_info remote_mem;
    rdma_client* thread_clients[MAX_RDMA_THREADS];
    void* thread_bufs[MAX_RDMA_THREADS];
    ibv_mr* thread_mrs[MAX_RDMA_THREADS];
    std::atomic<int> thread_count;
#endif

   public:
    std::atomic<uint64_t> read_count;
    std::atomic<uint64_t> write_count;

#ifdef PENTATHLON_USE_CXL
    RDMAClient(const sockaddr_storage&) {
        read_count.store(0);
        write_count.store(0);
        init_cxl_pool();
    }

    ~RDMAClient();
#else
    RDMAClient(const sockaddr_storage& addr) : addr_(addr) {
        read_count.store(0);
        write_count.store(0);
        thread_count.store(0);
        for (size_t i = 0; i < MAX_RDMA_THREADS; i++) {
            thread_clients[i] = nullptr;
            thread_bufs[i] = nullptr;
            thread_mrs[i] = nullptr;
        }
        // Match the main-branch RDMA client: pre-create one connection and
        // registered buffer for every supported worker thread.
        for (int i = 0; i < static_cast<int>(MAX_RDMA_THREADS); i++) {
            init_thread_resources(i);
        }
        init_thread_resources(0);
        remote_mem = thread_clients[0]->mem;
    }

    int get_thread_id() {
        static thread_local int thread_id = -1;
        if (thread_id < 0) thread_id = thread_count.fetch_add(1, std::memory_order_relaxed);
        if (thread_id >= static_cast<int>(MAX_RDMA_THREADS)) {
            throw std::runtime_error("too many RDMA threads");
        }
        return thread_id;
    }

    ~RDMAClient() {
        std::cout << "RDMA read: " << read_count << ", RDMA write: " << write_count << std::endl;
        // Clean up all thread buffers
        // rdma_client_free(inner);
    }
#endif

    memory_info mem() { return remote_mem; }

    void read(void* from, void* to, uint32_t size);
    void write(void* from, void* to, uint32_t size);

    void read_batch_add(void* from, void* to, uint32_t size);
    void write_batch_add(void* from, void* to, uint32_t size);
    void batch_commit();
};

#endif  // COMPUTE_RDMA_HPP
