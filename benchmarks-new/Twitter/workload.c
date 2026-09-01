#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "workload.h"

#define TWIT_ARENA_BLOCK_SIZE (1 << 20)
#define TWIT_KEY_MAX 256

#define TWIT_OP_UNSUPPORTED ((enum pth_twit_op)-1)

struct twit_arena {
    char **blocks;
    size_t block_count;
    size_t block_cap;
    size_t block_used;
};

struct twit_keymap_entry {
    const char *key;
    uint64_t id;
};

struct twit_keymap {
    struct twit_keymap_entry *entries;
    size_t capacity;
    size_t count;
};

static uint64_t twit_fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int twit_arena_init(struct twit_arena *a) {
    a->block_cap = 8;
    a->block_count = 0;
    a->block_used = TWIT_ARENA_BLOCK_SIZE;
    a->blocks = malloc(a->block_cap * sizeof(char *));
    if (!a->blocks) return -ENOMEM;
    return 0;
}

static int twit_arena_strdup(struct twit_arena *a, const char *s, const char **out) {
    size_t n = strlen(s) + 1;
    if (n > TWIT_ARENA_BLOCK_SIZE) return -ENOMEM;

    if (a->block_used + n > TWIT_ARENA_BLOCK_SIZE) {
        if (a->block_count == a->block_cap) {
            size_t new_cap = a->block_cap * 2;
            char **new_blocks = realloc(a->blocks, new_cap * sizeof(char *));
            if (!new_blocks) return -ENOMEM;
            a->blocks = new_blocks;
            a->block_cap = new_cap;
        }
        a->blocks[a->block_count] = malloc(TWIT_ARENA_BLOCK_SIZE);
        if (!a->blocks[a->block_count]) return -ENOMEM;
        a->block_count++;
        a->block_used = 0;
    }

    char *dst = a->blocks[a->block_count - 1] + a->block_used;
    memcpy(dst, s, n);
    a->block_used += n;
    *out = dst;
    return 0;
}

static void twit_arena_free(struct twit_arena *a) {
    for (size_t i = 0; i < a->block_count; i++) free(a->blocks[i]);
    free(a->blocks);
}

static int twit_keymap_init(struct twit_keymap *m, size_t cap) {
    size_t p = 1;
    while (p < cap) p <<= 1;
    if (p < 16) p = 16;
    m->capacity = p;
    m->count = 0;
    m->entries = calloc(m->capacity, sizeof(*m->entries));
    if (!m->entries) return -ENOMEM;
    return 0;
}

static int twit_keymap_grow(struct twit_keymap *m) {
    size_t new_cap = m->capacity << 1;
    struct twit_keymap_entry *new_entries = calloc(new_cap, sizeof(*new_entries));
    if (!new_entries) return -ENOMEM;

    size_t mask = new_cap - 1;
    for (size_t i = 0; i < m->capacity; i++) {
        if (!m->entries[i].key) continue;
        size_t j = twit_fnv1a(m->entries[i].key) & mask;
        while (new_entries[j].key) j = (j + 1) & mask;
        new_entries[j] = m->entries[i];
    }

    free(m->entries);
    m->entries = new_entries;
    m->capacity = new_cap;
    return 0;
}

static int twit_keymap_intern(struct twit_keymap *m, const char *key, uint64_t *out_id) {
    if ((m->count + 1) * 10 >= m->capacity * 7) {
        int rc = twit_keymap_grow(m);
        if (rc) return rc;
    }

    size_t mask = m->capacity - 1;
    size_t i = twit_fnv1a(key) & mask;
    while (m->entries[i].key) {
        if (strcmp(m->entries[i].key, key) == 0) {
            *out_id = m->entries[i].id;
            return 0;
        }
        i = (i + 1) & mask;
    }

    m->entries[i].key = key;
    m->entries[i].id = m->count;
    *out_id = m->count;
    m->count++;
    return 0;
}

static void twit_keymap_free(struct twit_keymap *m) {
    free(m->entries);
}

static enum pth_twit_op twit_parse_op(const char *s) {
    if (strcmp(s, "get") == 0) return TWIT_READ;
    if (strcmp(s, "set") == 0 || strcmp(s, "add") == 0) return TWIT_INSERT;
    if (strcmp(s, "delete") == 0) return TWIT_DELETE;
    return TWIT_OP_UNSUPPORTED;
}

static int twit_parse_trace(const char *path, struct twit_keymap *map, struct twit_arena *arena,
                            uint64_t warmup_num_req, uint64_t op_num_req, uint64_t bulk_load_num,
                            struct pth_twit_operation **out_warmup_ops, uint64_t *out_warmup_num,
                            struct pth_twit_operation **out_workload_ops,
                            uint64_t *out_workload_num) {
    FILE *f = fopen(path, "r");
    if (!f) {
        printf("Cannot open twitter trace: %s\n", path);
        return -errno;
    }

    uint64_t cap = warmup_num_req + op_num_req;
    if (cap < warmup_num_req) cap = 0;

    size_t arr_cap = 1 << 20;
    size_t n = 0;
    struct pth_twit_operation *all = malloc(arr_cap * sizeof(*all));
    if (!all) {
        fclose(f);
        return -ENOMEM;
    }

    unsigned long ts, ks, vs, cid, ttl;
    char key_buf[TWIT_KEY_MAX];
    char op_buf[32];

    while (fscanf(f, "%lu,%255[^,],%lu,%lu,%lu,%31[^,],%lu", &ts, key_buf, &ks, &vs, &cid,
                  op_buf, &ttl) == 7) {
        enum pth_twit_op op = twit_parse_op(op_buf);
        if (op == TWIT_OP_UNSUPPORTED) continue;

        const char *stored = NULL;
        int rc = twit_arena_strdup(arena, key_buf, &stored);
        if (rc) {
            free(all);
            fclose(f);
            return rc;
        }
        uint64_t id;
        rc = twit_keymap_intern(map, stored, &id);
        if (rc) {
            free(all);
            fclose(f);
            return rc;
        }

        /* The op array is bounded at warmup_num + op_num; keep interning keys
         * beyond that so the bulk-load-all default can enumerate every distinct
         * key without holding every op. */
        if (cap == 0 || n < cap) {
            if (n == arr_cap) {
                size_t new_cap = arr_cap << 1;
                struct pth_twit_operation *new_all = realloc(all, new_cap * sizeof(*all));
                if (!new_all) {
                    free(all);
                    fclose(f);
                    return -ENOMEM;
                }
                all = new_all;
                arr_cap = new_cap;
            }

            all[n].op = op;
            all[n].key = id;
            n++;
        }

        /* Early-stop I/O optimization: only when an explicit finite
         * PTH_BM_BULK_LOAD_NUM is set. When it is unset (TWIT_BULK_LOAD_ALL),
         * scan to EOF to enumerate all distinct keys for bulk-load. */
        if (bulk_load_num != TWIT_BULK_LOAD_ALL && cap != 0 &&
            map->count >= bulk_load_num && n >= cap) {
            break;
        }
    }
    fclose(f);

    uint64_t warmup_num = warmup_num_req;
    if (warmup_num > n) warmup_num = n;
    uint64_t op_num = op_num_req;
    if (op_num > n - warmup_num) op_num = n - warmup_num;

    struct pth_twit_operation *warmup_ops = NULL;
    struct pth_twit_operation *workload_ops = NULL;

    if (warmup_num) {
        warmup_ops = malloc(warmup_num * sizeof(*warmup_ops));
        if (!warmup_ops) {
            free(all);
            return -ENOMEM;
        }
        memcpy(warmup_ops, all, warmup_num * sizeof(*warmup_ops));
    }
    if (op_num) {
        workload_ops = malloc(op_num * sizeof(*workload_ops));
        if (!workload_ops) {
            free(warmup_ops);
            free(all);
            return -ENOMEM;
        }
        memcpy(workload_ops, all + warmup_num, op_num * sizeof(*workload_ops));
    }
    free(all);

    *out_warmup_ops = warmup_ops;
    *out_warmup_num = warmup_num;
    *out_workload_ops = workload_ops;
    *out_workload_num = op_num;
    return 0;
}

int pth_twit_parse(const struct pth_twit_workload_config *cfg, uint64_t *out_key_num,
                   struct pth_twit_operation **out_warmup_ops, uint64_t *out_warmup_num,
                   struct pth_twit_operation **out_workload_ops, uint64_t *out_workload_num) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", cfg->workload_dir, cfg->workload_name);

    struct twit_arena arena;
    int rc = twit_arena_init(&arena);
    if (rc) return rc;

    struct twit_keymap map;
    rc = twit_keymap_init(&map, 1 << 20);
    if (rc) {
        twit_arena_free(&arena);
        return rc;
    }

    rc = twit_parse_trace(path, &map, &arena, cfg->warmup_op_num, cfg->op_num, cfg->bulk_load_num,
                          out_warmup_ops, out_warmup_num, out_workload_ops, out_workload_num);
    if (rc) {
        twit_keymap_free(&map);
        twit_arena_free(&arena);
        return rc;
    }

    printf("Parsed %lu keys, %lu warmup ops, %lu workload ops from %s\n", map.count,
           *out_warmup_num, *out_workload_num, path);

    uint64_t key_num = map.count;
    twit_keymap_free(&map);
    twit_arena_free(&arena);

    if (out_key_num) *out_key_num = key_num;
    return 0;
}

void pth_twit_free_ops(struct pth_twit_operation *warmup_ops,
                       struct pth_twit_operation *workload_ops) {
    free(warmup_ops);
    free(workload_ops);
}
