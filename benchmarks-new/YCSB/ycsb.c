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
#include <numa.h>

#include "generators/xoshiro256p.h"
#include "target.h"
#include "workload.h"

struct ycsb_thread_context {
    struct pth_workload_context workload;
    char *bulk_load_array;
    uint64_t bulk_load_begin;
    uint64_t bulk_load_end;
};

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

static void run_ops(struct pth_workload_context *ctx, struct operation *ops, uint64_t op_num) {
    const struct pth_workload_config *cfg = ctx->config;

    for (uint64_t i = 0; i < op_num; i++) {
        double op = ops[i].op;
        uint64_t key = ops[i].key;

        if (op < cfg->read_ratio) {
            pth_bm_target_read(ctx->target, key);
        } else if (op < cfg->read_ratio + cfg->insert_ratio) {
            pth_bm_target_insert(ctx->target, key);
        } else if (op < cfg->read_ratio + cfg->insert_ratio + cfg->update_ratio) {
            pth_bm_target_update(ctx->target, key);
        } else {
            pth_bm_target_delete(ctx->target, key);
        }
    }
}

static void *workload_thread(void *data) {
    struct ycsb_thread_context *thread_ctx = data;
    struct pth_workload_context *ctx = &thread_ctx->workload;

    bind_core(ctx->cpuid);
    pth_bm_target_init_thread(ctx->target);

    for (uint64_t i = thread_ctx->bulk_load_begin; i < thread_ctx->bulk_load_end; ++i) {
        if (thread_ctx->bulk_load_array[i]) {
            pth_bm_target_insert(ctx->target, i);
        }
    }
    pth_bm_target_release_thread(ctx->target);
    atomic_fetch_add(&bulk_load_done, 1);

    while (!atomic_load(&warmup_ready))
        ;
    run_ops(ctx, ctx->warmup_ops, ctx->warmup_num);
    pth_bm_target_release_thread(ctx->target);
    atomic_fetch_add(&warmup_done, 1);

    while (!atomic_load(&workload_ready))
        ;
    run_ops(ctx, ctx->workload_ops, ctx->workload_num);
    atomic_fetch_add(&workload_done, 1);
    pth_bm_target_release_thread(ctx->target);

    return NULL;
}

static int setup_ctxs(const struct pth_workload_config *cfg, void *target, char *bulk_load_array,
                      uint64_t key_num, struct ycsb_thread_context **out_ctxs) {
    int rc;

    struct ycsb_thread_context *ctxs =
        calloc(cfg->thread_num, sizeof(struct ycsb_thread_context));
    if (!ctxs) {
        printf("Cannot allocate %d contexts\n", cfg->thread_num);
        rc = -ENOMEM;
        goto err;
    }

    struct pth_xoshiro256p master_uniform;
    uint64_t seed = 19260817;
    pth_xoshiro256p_init(&master_uniform, seed);
    printf("xoshiro256p master seed: %lu\n", seed);

    int cpuid = 1;

    uint64_t bulk_load_num_per_thread = key_num / cfg->thread_num;
    uint64_t bulk_load_num_rem = key_num % cfg->thread_num;
    uint64_t bulk_load_begin = 0;

    uint64_t warmup_num_per_thread = cfg->warmup_op_num / cfg->thread_num;
    uint64_t warmup_num_rem = cfg->warmup_op_num % cfg->thread_num;

    uint64_t workload_num_per_thread = cfg->op_num / cfg->thread_num;
    uint64_t workload_num_rem = cfg->op_num % cfg->thread_num;
    struct pth_workload_context global_ctx = {0};
    rc = pth_workload_context_init(&global_ctx, cfg, target, -1, &master_uniform,
                                   cfg->warmup_op_num, cfg->op_num);

    if (rc != 0) {
        printf("Cannot initialize workload context\n");
        goto err_free_ctxs;
    }

    for (int i = 0; i < cfg->thread_num; ++i) {
        uint64_t bulk_load_num = bulk_load_num_per_thread + !!(i < bulk_load_num_rem);
        uint64_t warmup_num = warmup_num_per_thread + !!(i < warmup_num_rem);
        uint64_t workload_num = workload_num_per_thread + !!(i < workload_num_rem);

        ctxs[i].bulk_load_array = bulk_load_array;
        ctxs[i].bulk_load_begin = bulk_load_begin;
        ctxs[i].bulk_load_end = bulk_load_begin + bulk_load_num;

        ctxs[i].workload.warmup_num = warmup_num;
        ctxs[i].workload.workload_num = workload_num;
        ctxs[i].workload.target = target;
        ctxs[i].workload.config = cfg;
        ctxs[i].workload.cpuid = cpuid;
        ctxs[i].workload.workload_ops = malloc(sizeof(struct operation) * workload_num);
        ctxs[i].workload.warmup_ops = malloc(sizeof(struct operation) * warmup_num);
        if (!ctxs[i].workload.workload_ops || !ctxs[i].workload.warmup_ops) {
            rc = -ENOMEM;
            printf("Cannot allocate workload operations for thread %d\n", i);
            for (int j = 0; j <= i; ++j) {
                free(ctxs[j].workload.workload_ops);
                free(ctxs[j].workload.warmup_ops);
            }
            free(global_ctx.workload_ops);
            free(global_ctx.warmup_ops);
            goto err_free_ctxs;
        }

        cpuid++;

        for (uint64_t j = 0; j < warmup_num; j++) {
            ctxs[i].workload.warmup_ops[j] = global_ctx.warmup_ops[j * cfg->thread_num + i];
        }
        for (uint64_t j = 0; j < workload_num; j++) {
            ctxs[i].workload.workload_ops[j] = global_ctx.workload_ops[j * cfg->thread_num + i];
        }

        bulk_load_begin = ctxs[i].bulk_load_end;
    }

    free(global_ctx.workload_ops);
    free(global_ctx.warmup_ops);

    *out_ctxs = ctxs;

    return 0;

err_free_ctxs:
    free(ctxs);
err:
    return rc;
}

static void shuffle(char *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t randnum = ((size_t)(rand() + rand()) << 32) + rand() + rand();
            size_t j = i + randnum / (UINT64_MAX / (n - i) + 1);
            char t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

static long get_long_from_env(const char *env_name, long default_value) {
    char *env = getenv(env_name);
    if (env) {
        return strtol(env, NULL, 10);
    } else {
        return default_value;
    }
}

static void free_ctxs(struct ycsb_thread_context *ctxs, int thread_num) {
    if (!ctxs) return;

    for (int i = 0; i < thread_num; ++i) {
        free(ctxs[i].workload.workload_ops);
        free(ctxs[i].workload.warmup_ops);
    }
    free(ctxs);
}

int main() {
    int rc = 0;

    bind_core(0);
    // numa_set_preferred(0);

    long key_num = get_long_from_env("PTH_BM_KEY_NUM", 200000000);
    long bulk_load_num = get_long_from_env("PTH_BM_BULK_LOAD_NUM", key_num / 2);
    struct pth_workload_config cfg = {
        .thread_num = get_long_from_env("PTH_BM_THREAD_NUM", 1),
        .key_num = key_num,
        .op_num = get_long_from_env("PTH_BM_OP_NUM", 100000000),
        .warmup_op_num = get_long_from_env("PTH_BM_WARMUP_OP_NUM", 10000000),
        .uniform = get_long_from_env("PTH_BM_UNIFORM", 0),
        .zipf_theta = 0.99,
        .read_ratio = get_long_from_env("PTH_BM_READ_RATIO", 95) / 100.0,
        .insert_ratio = get_long_from_env("PTH_BM_INSERT_RATIO", 5) / 100.0,
        .update_ratio = 0.0,
        .delete_ratio = 0.0,
    };

    printf("uniform %d\n", cfg.uniform);
    printf(
        "thread num %d, key num %lu, op num %lu, warmup op num %lu, bulk load num %ld, read ratio "
        "%f, insert ratio %f\n",
        cfg.thread_num, key_num, cfg.op_num, cfg.warmup_op_num, bulk_load_num, cfg.read_ratio,
        cfg.insert_ratio);

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
                        "uniform,thread_num,key_num,op_num,warmup_op_num,bulk_load_num,read_ratio,"
                        "insert_ratio,update_ratio,scan_ratio,duration\n");
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
    srand(19260817);

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

    // Target construction may access disaggregated memory on the main thread
    // (for example, to allocate a B+tree root).  The main thread does not take
    // part in the workload, so retire its cache/epoch registration before the
    // worker threads begin bulk loading.
    pth_bm_target_reset_data_manager();

    char *bulk_load_array = calloc(key_num, 1);
    if (!bulk_load_array) {
        printf("Cannot allocate bulk load array\n");
        rc = -ENOMEM;
        goto exit_destroy_target;
    }
    memset(bulk_load_array, 1, bulk_load_num);
    shuffle(bulk_load_array, key_num);

    struct ycsb_thread_context *ctxs = NULL;
    rc = setup_ctxs(&cfg, target, bulk_load_array, key_num, &ctxs);
    if (rc != 0) {
        printf("Cannot set up workload contexts\n");
        goto exit_free_bulk_load_array;
    }

    printf("Starting bulk load of %lu elements in range [0..%lu) ...\n", (uint64_t)bulk_load_num,
           (uint64_t)key_num);

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

    // pth_bm_target_reset_data_manager();

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

    if (file) {
        if (extra_cols && extra_col_values) fprintf(file, "%s,", extra_col_values);
        fprintf(file, "%d,%d,%ld,%lu,%lu,%ld,%lf,%lf,%lf,%lf,%lld\n", cfg.uniform, cfg.thread_num,
                key_num, cfg.op_num, cfg.warmup_op_num, bulk_load_num, cfg.read_ratio,
                cfg.insert_ratio, 0., 0., end - start);
        fclose(file);
    }

exit_free_ctxs:
    free_ctxs(ctxs, cfg.thread_num);
exit_free_bulk_load_array:
    free(bulk_load_array);
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
