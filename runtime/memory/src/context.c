#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "common/try.h"
#include "main.h"

#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

int memory_context_free(struct memory_context* ctx) {
    if (ctx->rdma) rdma_server_free(ctx->rdma);
    if (ctx->mem_pool) munmap(ctx->mem_pool, ctx->mem_pool_size);
    free(ctx);
    return 0;
}

struct memory_context* memory_context_create(uint16_t listen_port, size_t mem_pool_size) {
    struct memory_context* ctx = try2_p(calloc(1, sizeof(*ctx)));

    struct sockaddr_in6 addr = {.sin6_family = AF_INET6, .sin6_port = htons(listen_port)};

    size_t page_size = 2097152;  // TODO: get it from system
    ctx->mem_pool = try3_p(mmap(NULL, mem_pool_size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, 0, 0),
                           "failed to mmap");
    ctx->rdma =
        try3_p(rdma_server_create((struct sockaddr*)&addr, ctx->mem_pool, mem_pool_size, page_size),
               "failed to create RDMA server");
    return ctx;

cleanup:
    memory_context_free(ctx);
    return NULL;
}
