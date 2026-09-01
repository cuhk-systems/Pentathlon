#include <stdint.h>
#include <stdlib.h>

#include "workload.h"

int pth_workload_context_init(struct pth_workload_context *ctx,
                              const struct pth_workload_config *cfg, void *target, int cpuid,
                              struct pth_xoshiro256p *master_uniform, uint64_t warmup_num,
                              uint64_t workload_num) {
    ctx->target = target;
    ctx->config = cfg;
    ctx->cpuid = cpuid;

    ctx->uniform = *master_uniform;
    pth_xoshiro256p_jump(master_uniform);

    int rc = pth_zipf_init(&ctx->zipf, cfg->key_num, cfg->zipf_theta, master_uniform);
    if (rc != 0) return rc;
    pth_xoshiro256p_jump(master_uniform);

    ctx->warmup_num = warmup_num;
    ctx->warmup_ops = malloc(sizeof(struct operation) * warmup_num);
    for (size_t i = 0; i < warmup_num; i++) {
        ctx->warmup_ops[i].op = pth_xoshiro256p_next_f64(&ctx->uniform);
        if (cfg->uniform) {
            ctx->warmup_ops[i].key =
                ((double)pth_xoshiro256p_next(&ctx->uniform) / (double)UINT64_MAX) * cfg->key_num;
        } else {
            ctx->warmup_ops[i].key = pth_zipf_next(&ctx->zipf);
        }
    }
    ctx->workload_num = workload_num;
    ctx->workload_ops = malloc(sizeof(struct operation) * workload_num);
    for (size_t i = 0; i < workload_num; i++) {
        ctx->workload_ops[i].op = pth_xoshiro256p_next_f64(&ctx->uniform);
        if (cfg->uniform) {
            ctx->workload_ops[i].key =
                ((double)pth_xoshiro256p_next(&ctx->uniform) / (double)UINT64_MAX) * cfg->key_num;
        } else {
            ctx->workload_ops[i].key = pth_zipf_next(&ctx->zipf);
        }
    }

    return 0;
}
