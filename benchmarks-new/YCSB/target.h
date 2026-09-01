#ifndef PENTATHLON_BENCHMARKS_NEW_YCSB_TARGET_H_
#define PENTATHLON_BENCHMARKS_NEW_YCSB_TARGET_H_

void *pth_bm_target_create();
void pth_bm_target_init_thread(void *target);
void pth_bm_target_print_stat(void *target);
void pth_bm_target_destroy(void *target);

void pth_bm_target_read(void *target, int key);
void pth_bm_target_insert(void *target, int key);
void pth_bm_target_update(void *target, int key);
void pth_bm_target_delete(void *target, int key);
void pth_bm_target_scan(void *target, int from, int to);
void pth_bm_target_reset_data_manager();

#endif
