#ifndef PENTATHLON_BENCHMARKS_NEW_TWITTER_WORKLOAD_H_
#define PENTATHLON_BENCHMARKS_NEW_TWITTER_WORKLOAD_H_

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

/* Sentinel for "bulk-load all distinct keys" (PTH_BM_BULK_LOAD_NUM unset). */
#define TWIT_BULK_LOAD_ALL UINT64_MAX

enum pth_twit_op {
    TWIT_READ,
    TWIT_INSERT,
    TWIT_DELETE,
};

struct pth_twit_workload_config {
    int thread_num;
    uint64_t op_num;
    uint64_t warmup_op_num;
    uint64_t bulk_load_num;

    const char *workload_name;
    const char *workload_dir;

    double read_ratio;
    double write_ratio;
};

struct pth_twit_operation {
    enum pth_twit_op op;
    uint64_t key;
};

struct pth_twit_workload_context {
    void *target;
    const struct pth_twit_workload_config *config;
    int cpuid;

    uint64_t bulk_load_begin;
    uint64_t bulk_load_end;

    struct pth_twit_operation *warmup_ops;
    struct pth_twit_operation *workload_ops;
    uint64_t warmup_num;
    uint64_t workload_num;
};

int pth_twit_parse(const struct pth_twit_workload_config *cfg, uint64_t *out_key_num,
                   struct pth_twit_operation **out_warmup_ops, uint64_t *out_warmup_num,
                   struct pth_twit_operation **out_workload_ops, uint64_t *out_workload_num);

void pth_twit_free_ops(struct pth_twit_operation *warmup_ops, struct pth_twit_operation *workload_ops);

#endif
