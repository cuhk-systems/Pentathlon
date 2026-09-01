#include "lockfree_skiplist.hpp"

extern "C" {

void *pth_bm_target_create() { return new pth_skiplist::LockFreeSkipList(); }

void pth_bm_target_init_thread(void *target) {
    (void)target;
    pth_skiplist::LockFreeSkipList::seed_thread();
}

void pth_bm_target_destroy(void *target) {
    auto list = reinterpret_cast<pth_skiplist::LockFreeSkipList *>(target);
    delete list;
}

void pth_bm_target_read(void *target, int key) {
    auto list = reinterpret_cast<pth_skiplist::LockFreeSkipList *>(target);
    int result = 0;
    list->find(key, &result);
}

void pth_bm_target_insert(void *target, int key) {
    auto list = reinterpret_cast<pth_skiplist::LockFreeSkipList *>(target);
    list->insert(key, 0xbeef);
}

void pth_bm_target_update(void *target, int key) { pth_bm_target_insert(target, key); }

void pth_bm_target_delete(void *target, int key) {
    auto list = reinterpret_cast<pth_skiplist::LockFreeSkipList *>(target);
    list->erase(key);
}

void pth_bm_target_scan(void *target, int from, int to) {
    (void)target;
    (void)from;
    (void)to;
}

void __attribute__((weak)) pth_bm_target_reset_data_manager() {}

}
