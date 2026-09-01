#include <argp.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>

#include "common/handshake.h"
#include "common/parse.h"
#include "common/rdma-ops.h"
#include "main.h"
#include "rdma.h"

static const char args_doc[] = "ADDRESS";
static const char doc[] = "Memory server for the disaggregated memory compiler project";

static struct argp_option options[] = {
    {"verbose", 'v', NULL, 0, "Produce verbose output"},
    {"port", 'p', "PORT", 0, "Specify port to listen; defaults to 12345"},
    {},
};

static int parse_opt(int key, char* arg, struct argp_state* state) {
    struct arguments* args = state->input;
    switch (key) {
        case 'v':
            args->verbose = 1;
            break;
        case 'p':
            args->port_str = arg;
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

// TODO: properly do cleanup
int main(int argc, char** argv) {
    struct arguments args = {};
    argp_parse(&argp, argc, argv, 0, 0, &args);

    uint16_t port = args.port_str ? try(parse_port(args.port_str), "failed to parse port") : 12345;

    struct rlimit memlock = {.rlim_cur = RLIM_INFINITY, .rlim_max = RLIM_INFINITY};
    try(setrlimit(RLIMIT_MEMLOCK, &memlock), "failed to set rlimit");
    size_t mem_pool_size = (size_t)10 << 30; // 10 GiB

    struct memory_context* ctx =
        try_p(memory_context_create(port, mem_pool_size), "failed to initialize memory context");
    fprintf(stderr, "memory context initialized\n");

    int epfd = try(epoll_create1(0), "failed to create epoll fd");
    struct epoll_event ev, events[2];

    int rdma_events_fd = ctx->rdma->events->fd;
    ev = (typeof(ev)){.events = EPOLLIN, .data.fd = rdma_events_fd};
    try(epoll_ctl(epfd, EPOLL_CTL_ADD, rdma_events_fd, &ev), "failed to add fd to epoll");

    int ccfd = ctx->rdma->conn->cc->fd;
    ev = (typeof(ev)){.events = EPOLLIN, .data.fd = ccfd};
    try(epoll_ctl(epfd, EPOLL_CTL_ADD, ccfd, &ev), "failed to add fd to epoll");

    while (true) {
        int nfds = try(epoll_wait(epfd, events, 2, -1), "failed to epoll");
        for (int i = 0; i < nfds; i++) {
            if (events[i].data.fd == rdma_events_fd) {
                struct rdma_cm_event* rdma_event;
                try(rdma_get_cm_event(ctx->rdma->events, &rdma_event), "failed to get RDMA event");
                switch (rdma_event->event) {
                    case RDMA_CM_EVENT_DISCONNECTED:
                        fprintf(stderr, "compute connection disconnected\n");
                        rdma_ack_cm_event(rdma_event);
                        break;
                    case RDMA_CM_EVENT_ESTABLISHED:
                        // Connection requests and establishment notifications from
                        // different worker QPs may be interleaved.  Handle the
                        // notification in the event loop instead of synchronously
                        // waiting for it after each rdma_accept().
                        fprintf(stderr, "connection established\n");
                        rdma_ack_cm_event(rdma_event);
                        break;
                    case RDMA_CM_EVENT_CONNECT_REQUEST:
                        fprintf(stderr, "new connection\n");
                        struct rdma_cm_id* id = rdma_event->id;
                        struct rdma_conn_param param = rdma_event->param.conn;
                        rdma_ack_cm_event(rdma_event);

                        struct rdma_connection* conn = try3_p(
                            rdma_conn_create(id, true, ctx->rdma->conn->pd), "failed to create connection");

                        // struct ibv_mr* mem_pool_mr =
                        //     try3_p(ibv_reg_mr(conn->pd, ctx->mem_pool, ctx->mem_pool_size,
                        //                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        //                           IBV_ACCESS_REMOTE_WRITE),
                        //            "cannot register memory region");

                        // Accept parameters that compute side had proposed, plus memory-side
                        // handshake data (MR info, etc.)
                        struct memory_info hs = {
                            .addr = (uintptr_t)ctx->rdma->mem_pool_mr->addr,
                            .rkey = ctx->rdma->mem_pool_mr->rkey,
                            .page_size = 2097152,
                            .page_count = mem_pool_size / 2097152,
                        };
                        param.private_data = &hs;
                        param.private_data_len = sizeof(hs);

                        try3(rdma_accept(id, &param), "failed to accept");
                        break;
                    cleanup:
                        exit(1);
                    default:
                        fprintf(stderr, "received new RDMA event %s\n",
                                rdma_event_str(rdma_event->event));
                        rdma_ack_cm_event(rdma_event);
                        break;
                }

            } else if (events[i].data.fd == ccfd) {
                struct rdma_connection* c = ctx->rdma->conn;
                struct ibv_wc wcs[MAX_POLL];
                int polled = try(rdma_conn_poll_ev(c, wcs, MAX_POLL), "failed to poll");

                bool errored = false;
                for (int i = 0; i < polled; i++) {
                    if (wcs[i].status != IBV_WC_SUCCESS) {
                        fprintf(stderr, "recv completion queue received error: %s\n",
                                ibv_wc_status_str(wcs[i].status));
                        errored = true;
                    }
                    if (!errored) {
                        // refill recv queue
                        // currently we only have 1 slot of recv buffer; will use wr_id to indicate
                        // which slot to refill.
                        try(rdma_post_recv(c->id, NULL, c->recv_buf, sizeof(*c->recv_buf),
                                           c->recv_mr),
                            "failed to RDMA recv");
                    }
                }
                if (errored) return -1;

            } else {
                fprintf(stderr, "unknown fd %d in epoll\n", events[i].data.fd);
                return -1;
            }
        }
    }

    try(memory_context_free(ctx), "failed to free memory context");
    return 0;
}
