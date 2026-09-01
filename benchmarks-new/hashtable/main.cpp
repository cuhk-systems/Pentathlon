#include "lockfree_hashtable.hpp"

#include <cstdio>

extern "C" {

void *pth_bm_target_create() { return new pth_hashtable::LockFreeHashTable(); }

void pth_bm_target_destroy(void *target) {
    auto table = reinterpret_cast<pth_hashtable::LockFreeHashTable *>(target);
    delete table;
}

void pth_bm_target_read(void *target, int key) {
    auto table = reinterpret_cast<pth_hashtable::LockFreeHashTable *>(target);
    int result = 0;
    table->get(key, &result);
}

void pth_bm_target_insert(void *target, int key) {
    auto table = reinterpret_cast<pth_hashtable::LockFreeHashTable *>(target);
    table->put(key, 0xbeef);
}

void pth_bm_target_update(void *target, int key) { pth_bm_target_insert(target, key); }

void pth_bm_target_delete(void *target, int key) {
    auto table = reinterpret_cast<pth_hashtable::LockFreeHashTable *>(target);
    table->erase(key);
}

void pth_bm_target_scan(void *target, int from, int to) {
    (void)target;
    (void)from;
    (void)to;
}

void __attribute__((weak)) pth_bm_target_reset_data_manager() {}

void __attribute__((weak)) pth_bm_target_print_stat(void *target) {
    auto table = reinterpret_cast<pth_hashtable::LockFreeHashTable *>(target);
    printf("hashtable buckets %zu, live items %llu\n",
           pth_hashtable::LockFreeHashTable::bucket_count(),
           static_cast<unsigned long long>(table->size()));
}

}
