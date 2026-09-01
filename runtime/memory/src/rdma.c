#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <stdio.h>
#include <stdlib.h>

#include "common/handshake.h"
#include "common/rdma-ops.h"
#include "common/rdma.h"
#include "main.h"

int rdma_server_free(struct rdma_server* server) {
    if (server->conn) rdma_conn_free(server->conn);
    if (server->mem_pool_mr) ibv_dereg_mr(server->mem_pool_mr);
    if (server->listen_id) rdma_destroy_id(server->listen_id);
    if (server->events) rdma_destroy_event_channel(server->events);
    return 0;
}

// TODO: pass MR information to compute side
struct rdma_server* rdma_server_create(struct sockaddr* addr, void* mem_pool, size_t mem_pool_size,
                                       size_t page_size) {
    struct rdma_server* server = try2_p(calloc(1, sizeof(*server)));
    server->events = try3_p(rdma_create_event_channel(), "failed to create RDMA event channel");
    try3(rdma_create_id(server->events, &server->listen_id, NULL, RDMA_PS_TCP),
         "failed to create listen ID");

    // Bind and listen for client to connect
    try3(rdma_bind_addr(server->listen_id, addr), "failed to bind address");
    // fprintf(stderr, "server is listening at %s:%d\n", inet_ntoa(addr->sin_addr),
    //         ntohs(addr.sin_port));
    fprintf(stderr, "Waiting for connection from compute...\n");
    // Each compute worker owns a QP, so allow a full benchmark's connection
    // burst to queue while the memory server accepts them.
    try3(rdma_listen(server->listen_id, 128), "failed to listen");

    // Get CM ID for the connection
    struct rdma_cm_id* conn_id = NULL;
    struct rdma_conn_param param = {};
    try3(expect_connect_request(server->events, &conn_id, &param), "failed");
    server->conn = try3_p(rdma_conn_create(conn_id, true, NULL), "failed to create connection");

    server->mem_pool_mr = try3_p(
        ibv_reg_mr(server->conn->pd, mem_pool, mem_pool_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE),
        "cannot register memory region");

    // Accept parameters that compute side had proposed, plus memory-side handshake data (MR info,
    // etc.)
    struct memory_info hs = {
        .addr = (uintptr_t)server->mem_pool_mr->addr,
        .rkey = server->mem_pool_mr->rkey,
        .page_size = page_size,
        .page_count = mem_pool_size / page_size,
    };
    param.private_data = &hs;
    param.private_data_len = sizeof(hs);

    try3(rdma_accept(server->conn->id, &param), "failed to accept");
    try3(expect_established(server->events, NULL));

    return server;

cleanup:
    rdma_server_free(server);
    return NULL;
}
