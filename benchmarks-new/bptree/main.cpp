#include "btree_local.hpp"

extern "C" {
void *pth_bm_target_create() { return new btreeolc::BTree<int, int>(); }
void pth_bm_target_destroy(void *target) {
    auto tree = reinterpret_cast<btreeolc::BTree<int, int> *>(target);
    delete tree;
}
void pth_bm_target_read(void *target, int key) {
    auto tree = reinterpret_cast<btreeolc::BTree<int, int> *>(target);
    int result = 0;
    tree->lookup(key, result);
}
void pth_bm_target_insert(void *target, int key) {
    auto tree = reinterpret_cast<btreeolc::BTree<int, int> *>(target);
    tree->insert(key, 0xbeef);
}
void pth_bm_target_update(void *target, int key) { pth_bm_target_insert(target, key); }
void pth_bm_target_delete(void *target, int key) {}
}
