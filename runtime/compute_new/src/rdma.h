#ifndef COMPUTE_RDMA_H
#define COMPUTE_RDMA_H

#include <rdma/rdma_cma.h>

#include "common/handshake.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rdma_client {
    struct rdma_event_channel* rdma_events;
    struct rdma_connection* conn;
    struct memory_info mem;
};

struct rdma_client* rdma_client_connect(struct sockaddr* addr);
int rdma_client_free(struct rdma_client* client);

// reuse C function here
int c_parse_addr(char* addr_str, struct sockaddr_storage* addr, int default_port);

#ifdef __cplusplus
}
#endif

#endif  // COMPUTE_RDMA_H
