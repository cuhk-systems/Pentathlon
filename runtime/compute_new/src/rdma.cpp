#include <alloca.h>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rdma.hpp"

// Thread-safe RDMA implementation using per-thread buffers
namespace {
void post_send_and_wait(rdma_connection* conn, ibv_send_wr* wr) {
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(conn->id->qp, wr, &bad)) {
        throw std::runtime_error("failed to post send");
    }
    if (bad) {
        throw std::runtime_error("bad wr pointer non null");
    }

    ibv_wc wc;
    while (true) {
        int ret = ibv_poll_cq(conn->send_cq, 1, &wc);
        if (ret == 1) break;
        if (ret < 0) {
            throw std::runtime_error("failed to poll completion queue");
        }
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error("work completion error");
    }
}
}  // namespace

void RDMAClient::init_thread_resources(int id) {
    if (thread_clients[id]) {
        return;
    }

    thread_clients[id] = rdma_client_connect(reinterpret_cast<sockaddr*>(&addr_));
    if (!thread_clients[id]) {
        throw std::runtime_error("failed to create RDMA client");
    }

    thread_bufs[id] = malloc(CACHE_LINE_SIZE * MAX_BATCH_SIZE);
    assert(thread_bufs[id]);
    thread_mrs[id] = ibv_reg_mr(thread_clients[id]->conn->pd, thread_bufs[id],
                                CACHE_LINE_SIZE * MAX_BATCH_SIZE, IBV_ACCESS_LOCAL_WRITE);
    assert(thread_mrs[id]);
}

void RDMAClient::ensure_thread_resources(int id) {
    if (id < 0 || id >= static_cast<int>(MAX_RDMA_THREADS)) {
        throw std::runtime_error("invalid RDMA thread id");
    }
    init_thread_resources(id);
}

void RDMAClient::read(void* from, void* to, uint32_t size) {
    assert(size <= CACHE_LINE_SIZE);

    ThreadBuffer& thread_buf = get_thread_buffer();
    const uint32_t rkey = remote_mem.rkey;

    ibv_sge sg = {
        .addr = (uint64_t)thread_buf.buf, .length = size, .lkey = thread_buf.buf_mr->lkey};
    ibv_send_wr wr = {};
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr = {.rdma = {.remote_addr = reinterpret_cast<uint64_t>(from), .rkey = rkey}};

    post_send_and_wait(thread_buf.client->conn, &wr);

    memcpy(to, thread_buf.buf, size);

    read_count.fetch_add(1, std::memory_order_relaxed);
}

void RDMAClient::write(void* from, void* to, uint32_t size) {
    assert(size <= CACHE_LINE_SIZE);

    ThreadBuffer& thread_buf = get_thread_buffer();
    const uint32_t rkey = remote_mem.rkey;
    memcpy(thread_buf.buf, from, size);

    ibv_sge sg = {
        .addr = (uint64_t)thread_buf.buf, .length = size, .lkey = thread_buf.buf_mr->lkey};
    ibv_send_wr wr = {};
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr = {.rdma = {.remote_addr = reinterpret_cast<uint64_t>(to), .rkey = rkey}};

    post_send_and_wait(thread_buf.client->conn, &wr);

    write_count.fetch_add(1, std::memory_order_relaxed);
}

void RDMAClient::read_batch_add(void* from, void* to, uint32_t size) {
    assert(size <= CACHE_LINE_SIZE);
    auto& thread_buf = get_thread_buffer();
    auto& batch = ThreadBatch::get_instance();
    const uint32_t rkey = remote_mem.rkey;

    if (batch.count == MAX_BATCH_SIZE) batch_commit();
    auto& [wr, sg] = batch.get_next();
    auto buf = (char*)thread_buf.buf + CACHE_LINE_SIZE * (batch.count - 1);

    sg.addr = (uint64_t)buf;
    sg.length = size;
    sg.lkey = thread_buf.buf_mr->lkey;

    wr.opcode = IBV_WR_RDMA_READ;
    wr.wr = {.rdma = {.remote_addr = (uint64_t)from, .rkey = rkey}};

    batch.push_copy_after_read(buf, to, size);
    batch.read_count++;
}

void RDMAClient::write_batch_add(void* from, void* to, uint32_t size) {
    assert(size <= CACHE_LINE_SIZE);
    auto& thread_buf = get_thread_buffer();
    auto& batch = ThreadBatch::get_instance();
    const uint32_t rkey = remote_mem.rkey;

    if (batch.count == MAX_BATCH_SIZE) batch_commit();
    auto& [wr, sg] = batch.get_next();
    auto buf = (char*)thread_buf.buf + CACHE_LINE_SIZE * (batch.count - 1);
    memcpy(buf, from, size);

    sg.addr = (uint64_t)buf;
    sg.length = size;
    sg.lkey = thread_buf.buf_mr->lkey;

    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr = {.rdma = {.remote_addr = (uint64_t)to, .rkey = rkey}};

    batch.write_count++;
}

void RDMAClient::batch_commit() {
    auto& thread_buf = get_thread_buffer();
    auto& batch = ThreadBatch::get_instance();
    if (batch.count == 0) return;
    batch.wrs[batch.count - 1].first.send_flags |= IBV_SEND_SIGNALED;

    post_send_and_wait(thread_buf.client->conn, &batch.wrs[0].first);

    for (size_t i = 0; i < batch.copy_after_read_count; i++) {
        auto& [from, to, size] = batch.copy_after_read[i];
        memcpy(to, from, size);
    }

    read_count.fetch_add(batch.read_count, std::memory_order_relaxed);
    write_count.fetch_add(batch.write_count, std::memory_order_relaxed);
    batch.reset();
}
