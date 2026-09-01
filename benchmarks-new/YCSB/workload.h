#ifndef PENTATHLON_BENCHMARKS_NEW_YCSB_WORKLOAD_H_
#define PENTATHLON_BENCHMARKS_NEW_YCSB_WORKLOAD_H_

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "generators/xoshiro256p.h"
#include "generators/zipf.h"

struct pth_workload_config {
    int thread_num;
    uint64_t key_num;
    uint64_t warmup_op_num;
    uint64_t op_num;

    bool uniform;
    double zipf_theta;

    double read_ratio;
    double insert_ratio;
    double update_ratio;
    double delete_ratio;
};

struct pth_workload_context {
    void *target;
    const struct pth_workload_config *config;
    int cpuid;

    struct pth_xoshiro256p uniform;
    struct pth_zipf zipf;

    struct operation {
        double op;
        uint64_t key;
    } *warmup_ops, *workload_ops;
    uint64_t warmup_num, workload_num;
};

int pth_workload_context_init(struct pth_workload_context *ctx,
                              const struct pth_workload_config *cfg, void *target, int cpuid,
                              struct pth_xoshiro256p *master_uniform, uint64_t warmup_num,
                              uint64_t workload_num);

#endif
