#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "target.h"
#include "workload.h"

#ifndef TWIT_WORKLOAD_DIR
#define TWIT_WORKLOAD_DIR "."
#endif

static long long get_real_time() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        return -1;
    }
    return ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

void __attribute__((weak)) pth_bm_target_init_thread(void *target) {}
void __attribute__((weak)) pth_bm_target_print_stat(void *target) {}
void __attribute__((weak)) pth_bm_target_release_thread(void *target) {}

static void bind_core(int cpuid) {
    if (cpuid < 0) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);

    pthread_t thread = pthread_self();
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc < 0) {
        perror("pthread_setaffinity_np");
    }
}

static _Atomic int bulk_load_done = 0;
static _Atomic bool warmup_ready = false;
static _Atomic int warmup_done = 0;
static _Atomic bool workload_ready = false;
static _Atomic int workload_done = 0;

static void run_ops(struct pth_twit_workload_context *ctx, struct pth_twit_operation *ops,
                    uint64_t op_num) {
    for (uint64_t i = 0; i < op_num; i++) {
        switch (ops[i].op) {
            case TWIT_READ:
                pth_bm_target_read(ctx->target, ops[i].key);
                break;
            case TWIT_INSERT:
                pth_bm_target_insert(ctx->target, ops[i].key);
                break;
            case TWIT_DELETE:
                pth_bm_target_delete(ctx->target, ops[i].key);
                break;
        }
    }
}

static void *workload_thread(void *data) {
    struct pth_twit_workload_context *ctx = data;

    bind_core(ctx->cpuid);
    pth_bm_target_init_thread(ctx->target);

    for (uint64_t i = ctx->bulk_load_begin; i < ctx->bulk_load_end; ++i) {
        pth_bm_target_insert(ctx->target, i);
    }
    atomic_fetch_add(&bulk_load_done, 1);

    while (!atomic_load(&warmup_ready))
        ;
    run_ops(ctx, ctx->warmup_ops, ctx->warmup_num);
    atomic_fetch_add(&warmup_done, 1);

    while (!atomic_load(&workload_ready))
        ;
    run_ops(ctx, ctx->workload_ops, ctx->workload_num);
    atomic_fetch_add(&workload_done, 1);
    pth_bm_target_release_thread(ctx->target);

    return NULL;
}

static int setup_ctxs(const struct pth_twit_workload_config *cfg, void *target,
                      struct pth_twit_operation *global_warmup_ops, uint64_t warmup_total,
                      struct pth_twit_operation *global_workload_ops, uint64_t workload_total,
                      struct pth_twit_workload_context **out_ctxs) {
    struct pth_twit_workload_context *ctxs =
        calloc(cfg->thread_num, sizeof(struct pth_twit_workload_context));
    if (!ctxs) {
        printf("Cannot allocate %d contexts\n", cfg->thread_num);
        return -ENOMEM;
    }

    int cpuid = 2;

    uint64_t bulk_per_thread = cfg->bulk_load_num / cfg->thread_num;
    uint64_t bulk_rem = cfg->bulk_load_num % cfg->thread_num;
    uint64_t bulk_begin = 0;

    uint64_t warmup_per_thread = warmup_total / cfg->thread_num;
    uint64_t warmup_rem = warmup_total % cfg->thread_num;

    uint64_t workload_per_thread = workload_total / cfg->thread_num;
    uint64_t workload_rem = workload_total % cfg->thread_num;

    for (int i = 0; i < cfg->thread_num; ++i) {
        uint64_t bulk_num = bulk_per_thread + !!(i < bulk_rem);
        uint64_t warmup_num = warmup_per_thread + !!(i < warmup_rem);
        uint64_t workload_num = workload_per_thread + !!(i < workload_rem);

        ctxs[i].bulk_load_begin = bulk_begin;
        ctxs[i].bulk_load_end = bulk_begin + bulk_num;
        bulk_begin = ctxs[i].bulk_load_end;

        ctxs[i].target = target;
        ctxs[i].config = cfg;
        ctxs[i].cpuid = cpuid;
        ctxs[i].warmup_num = warmup_num;
        ctxs[i].workload_num = workload_num;
        ctxs[i].warmup_ops = malloc(sizeof(struct pth_twit_operation) * warmup_num);
        ctxs[i].workload_ops = malloc(sizeof(struct pth_twit_operation) * workload_num);
        if (!ctxs[i].warmup_ops || !ctxs[i].workload_ops) {
            printf("Cannot allocate operations for thread %d\n", i);
            for (int j = 0; j <= i; ++j) {
                free(ctxs[j].warmup_ops);
                free(ctxs[j].workload_ops);
            }
            free(ctxs);
            return -ENOMEM;
        }

        cpuid += 2;
        if (cpuid >= 40) cpuid = 1;

        for (uint64_t j = 0; j < warmup_num; j++) {
            ctxs[i].warmup_ops[j] = global_warmup_ops[j * cfg->thread_num + i];
        }
        for (uint64_t j = 0; j < workload_num; j++) {
            ctxs[i].workload_ops[j] = global_workload_ops[j * cfg->thread_num + i];
        }
    }

    *out_ctxs = ctxs;
    return 0;
}

static long get_long_from_env(const char *env_name, long default_value) {
    char *env = getenv(env_name);
    if (env) {
        return strtol(env, NULL, 10);
    } else {
        return default_value;
    }
}

static void free_ctxs(struct pth_twit_workload_context *ctxs, int thread_num) {
    if (!ctxs) return;

    for (int i = 0; i < thread_num; ++i) {
        free(ctxs[i].warmup_ops);
        free(ctxs[i].workload_ops);
    }
    free(ctxs);
}

int main() {
    int rc = 0;

    bind_core(0);

    const char *workload_name = getenv("PTH_BM_WORKLOAD");
    if (!workload_name) workload_name = "cluster3.sort";

    uint64_t bulk_load_num;
    const char *bulk_load_env = getenv("PTH_BM_BULK_LOAD_NUM");
    if (bulk_load_env) {
        long v = strtol(bulk_load_env, NULL, 10);
        if (v < 0) v = 0;
        bulk_load_num = (uint64_t)v;
    } else {
        bulk_load_num = TWIT_BULK_LOAD_ALL;
    }

    struct pth_twit_workload_config cfg = {
        .thread_num = get_long_from_env("PTH_BM_THREAD_NUM", 1),
        .op_num = get_long_from_env("PTH_BM_OP_NUM", 6800000),
        .warmup_op_num = get_long_from_env("PTH_BM_WARMUP_OP_NUM", 1000000),
        .bulk_load_num = bulk_load_num,
        .workload_name = workload_name,
        .workload_dir = TWIT_WORKLOAD_DIR,
        .read_ratio = 0.0,
        .write_ratio = 0.0,
    };

    printf("twitter trace workload %s, dir %s, thread num %d, warmup op num %lu, op num %lu\n",
           workload_name, TWIT_WORKLOAD_DIR, cfg.thread_num, cfg.warmup_op_num, cfg.op_num);
    if (cfg.bulk_load_num == TWIT_BULK_LOAD_ALL)
        printf("bulk load num: all keys (default)\n");
    else
        printf("bulk load num: %lu\n", cfg.bulk_load_num);

    const char *filename = getenv("PTH_BM_FILENAME");
    const char *extra_cols = getenv("PTH_BM_EXTRA_COLS");
    const char *extra_col_values = getenv("PTH_BM_EXTRA_COL_VALUES");
    FILE *file = NULL;
    if (filename) {
        printf("result will be saved to %s\n", filename);
        if (access(filename, F_OK) == -1) {
            file = fopen(filename, "a");
            if (file) {
                if (extra_cols && extra_col_values) fprintf(file, "%s,", extra_cols);
                fprintf(file,
                        "workload,thread_num,op_num,warmup_op_num,bulk_load_num,read_ratio,"
                        "write_ratio,duration\n");
                fflush(file);
            } else {
                perror("Error creating file");
                rc = 1;
                goto exit;
            }
        } else {
            file = fopen(filename, "a");
        }
    } else {
        printf("no filename given, report not saved\n");
    }
    if (extra_cols && extra_col_values)
        printf("extra columns: (%s) = (%s)\n", extra_cols, extra_col_values);

    bool cancelled = false;

    pthread_t *thds = malloc(cfg.thread_num * sizeof(pthread_t));
    if (!thds) {
        printf("Cannot allocate workload thread list\n");
        rc = -ENOMEM;
        goto exit;
    }

    void *target = pth_bm_target_create();
    if (!target) {
        printf("Benchmark implementation returned a NULL pointer\n");
        rc = -EINVAL;
        goto exit_free_thds;
    }

    struct pth_twit_operation *global_warmup_ops = NULL;
    struct pth_twit_operation *global_workload_ops = NULL;
    uint64_t warmup_total = 0;
    uint64_t workload_total = 0;
    uint64_t key_num = 0;

    rc = pth_twit_parse(&cfg, &key_num, &global_warmup_ops, &warmup_total, &global_workload_ops,
                        &workload_total);
    if (rc != 0) {
        printf("Cannot parse twitter workload\n");
        goto exit_destroy_target;
    }

    if (cfg.bulk_load_num == TWIT_BULK_LOAD_ALL)
        cfg.bulk_load_num = key_num;
    else if (cfg.bulk_load_num > key_num)
        cfg.bulk_load_num = key_num;

    uint64_t wl_read = 0;
    uint64_t wl_write = 0;
    for (uint64_t i = 0; i < workload_total; i++) {
        if (global_workload_ops[i].op == TWIT_READ)
            wl_read++;
        else
            wl_write++;
    }
    cfg.read_ratio = workload_total ? (double)wl_read / workload_total : 0.0;
    cfg.write_ratio = workload_total ? (double)wl_write / workload_total : 0.0;
    printf("workload ratios: read %f, write %f\n", cfg.read_ratio, cfg.write_ratio);

    struct pth_twit_workload_context *ctxs = NULL;
    rc = setup_ctxs(&cfg, target, global_warmup_ops, warmup_total, global_workload_ops,
                    workload_total, &ctxs);
    if (rc != 0) {
        printf("Cannot set up workload contexts\n");
        goto exit_free_ops;
    }

    pth_twit_free_ops(global_warmup_ops, global_workload_ops);
    global_warmup_ops = NULL;
    global_workload_ops = NULL;

    printf("Starting bulk load of %lu keys ...\n", cfg.bulk_load_num);

    for (int tcount = 0; tcount < cfg.thread_num; ++tcount) {
        rc = pthread_create(&thds[tcount], NULL, workload_thread, &ctxs[tcount]);
        if (rc != 0) {
            printf("Cannot create workload thread %d\n", tcount);

            for (int i = 0; i < tcount; ++i) pthread_cancel(thds[i]);

            cancelled = true;
            goto exit_free_ctxs;
        }
    }

    while (atomic_load(&bulk_load_done) < cfg.thread_num)
        ;

    printf("Bulk load finished, starting warmup...\n");

    sleep(1);
    atomic_store(&warmup_ready, true);
    while (atomic_load(&warmup_done) < cfg.thread_num)
        ;

    pth_bm_target_print_stat(target);
    printf("Warmup done, starting benchmark...\n");

    sleep(1);
    atomic_store(&workload_ready, true);
    long long start = get_real_time();
    while (atomic_load(&workload_done) < cfg.thread_num)
        ;
    long long end = get_real_time();

    for (int i = 0; i < cfg.thread_num; ++i) {
        pthread_join(thds[i], NULL);
    }

    if (!cancelled) {
        printf("Total duration: %lld\n", end - start);
    }

    pth_bm_target_print_stat(target);

    if (!cancelled && file) {
        if (extra_cols && extra_col_values) fprintf(file, "%s,", extra_col_values);
        fprintf(file, "%s,%d,%lu,%lu,%lu,%lf,%lf,%lld\n", workload_name, cfg.thread_num, cfg.op_num,
                cfg.warmup_op_num, cfg.bulk_load_num, cfg.read_ratio, cfg.write_ratio, end - start);
        fclose(file);
    }

exit_free_ctxs:
    free_ctxs(ctxs, cfg.thread_num);
exit_free_ops:
    pth_twit_free_ops(global_warmup_ops, global_workload_ops);
exit_destroy_target:
    // pth_bm_target_destroy(target);
exit_free_thds:
    free(thds);
exit:
    if (rc == 0)
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}
