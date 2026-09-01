// Initial TCP control plane before RDMA

#ifndef _COMMON_RDMA_H_
#define _COMMON_RDMA_H_

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "try.h"

// Wait for a certain type of CM event, regarding others as error.
static inline int _expect_event(struct rdma_event_channel* events, enum rdma_cm_event_type type,
                                struct rdma_cm_id** conn_id, struct rdma_conn_param* param) {
  struct rdma_cm_event* ev;
  try(rdma_get_cm_event(events, &ev), "cannot get CM event");
  if (ev->event != type) {
    fprintf(stderr, "expected %s, got %s\n", rdma_event_str(type), rdma_event_str(ev->event));
    rdma_ack_cm_event(ev);
    return -1;
  }
  fprintf(stderr, "ayy!! %s\n", rdma_event_str(type));
  if (conn_id) *conn_id = ev->id;
  if (param) *param = ev->param.conn;
  rdma_ack_cm_event(ev);
  return 0;
}

static inline int expect_event(struct rdma_event_channel* events, enum rdma_cm_event_type type) {
  return _expect_event(events, type, NULL, NULL);
}

static inline int expect_connect_request(struct rdma_event_channel* events,
                                         struct rdma_cm_id** conn_id,
                                         struct rdma_conn_param* param) {
  return _expect_event(events, RDMA_CM_EVENT_CONNECT_REQUEST, conn_id, param);
}

static inline int expect_established(struct rdma_event_channel* events,
                                     struct rdma_conn_param* param) {
  return _expect_event(events, RDMA_CM_EVENT_ESTABLISHED, NULL, param);
}

// Current version seems to only need RDMA read and write, without control messages. Reserved for
// future use.
struct message {
  uint8_t _reserved[16];
};

//
struct rdma_connection {
  struct rdma_cm_id* id;  // RDMA communication manager ID, like socket
  struct ibv_pd* pd;      // protection domain

  struct ibv_comp_channel* cc;       // receive completion channel, for event notification
  struct ibv_cq *send_cq, *recv_cq;  // completion queues

  struct message *send_buf, *recv_buf;  // message buffers
  struct ibv_mr *send_mr, *recv_mr;     // message buffers' registered memory regions
};

#endif
