#ifndef COMPUTE_DATA_MANAGER_H
#define COMPUTE_DATA_MANAGER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <atomic>
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

#include "addr.h"
#include "lru.hpp"
#include "metadata.hpp"
#include "rdma.hpp"
// #include "epochmanager.hpp"

class DataManager {
    RDMAClient* rdma;

    std::atomic<uint64_t> chunk_addr;
    std::atomic<size_t> cache_size;
    // Memory allocation
    static constexpr size_t MAX_SIZE = 1 << 14;
    // max size: 16KB
    void* cache_pool;
    std::atomic<size_t> cache_pool_offset;
    std::atomic<bool> assignFinished{false};

    uint64_t compute_cache_limit;
    // uint64_t compute_page_size;

    size_t chunk_size() { return rdma->mem().page_size; }

    const uint8_t state_not_in_cache = 0;
    const uint8_t state_in_cache_reclaiming = 1;
    const uint8_t state_in_cache_cooling = 2;
    const uint8_t state_in_cache_hot = 3;

    inline bool is_cache_full(size_t size) {
        // Check if the cache size exceeds the limit
        return cache_size.load() + size > compute_cache_limit;
    }

    bool try_evict_hot_leaf(MetaData* metadata);
    void ensure_epoch_registered(int tid);
    void publish_quiescent_epoch(int tid);
    void retire_cooling_addr(int tid, MetaData* metadata, void* local_addr, size_t size);
    void reclaim_cooling_addrs(int tid);
    bool can_reclaim_epoch(uint64_t retire_epoch);
    bool should_cache_cold_node(MetaData* metadata);
    void on_node_became_hot(MetaData* metadata, MetaDataAddr* metadata_addr, int tid);

   public:

    std::atomic<uint64_t> malloc_size;
    std::atomic<uint64_t> malloc_count;
    std::atomic<uint64_t> thread_count;
    // EpochManager epoch_manager;
    inline int get_thread_id() {
        static thread_local int thread_id = -1;
        if (thread_id < 0) thread_id = thread_count.fetch_add(1);
        return thread_id;
    }
    constexpr static size_t MAX_THREADS = kComputeMaxThreads;
    constexpr static size_t STRIDE = 16;

    struct CoolingRetireEntry {
        MetaData* metadata;
        void* local_addr;
        size_t size;
        uint64_t retire_epoch;
    };

    struct alignas(64) ThreadEpochState {
        std::atomic<uint64_t> epoch;
        std::atomic<bool> active;

        ThreadEpochState() : epoch(0), active(false) {}
    };

    OwnerLruManager owner_lru;
    std::vector<void*> evicted_addrs[MAX_THREADS];
    std::vector<CoolingRetireEntry> cooling_addrs[MAX_THREADS];
    alignas(64) size_t cooling_reclaim_head[MAX_THREADS * STRIDE];
    std::array<ThreadEpochState, MAX_THREADS * STRIDE> epoch_states;
    alignas(64) std::atomic<MetaData*> last_cxl_access[MAX_THREADS * STRIDE];
    alignas(64) std::atomic<uint64_t> global_epoch{1};
    uint64_t cold_cache_frequency_ns{1000000};
    uint64_t cache_admit_threshold{4};

    void disaggFree(GlobalAddr gaddr);
    GlobalAddr disaggAlloc(size_t size);
    void addAddrDep(GlobalAddr addr_u, GlobalAddr addr_v);  // u -> v
    void* updateAddrDep(GlobalAddr father, GlobalAddr child);
    void* getLocalAddr(GlobalAddr gaddr);
    void* assignLocalAddr(MetaData* metadata, bool read);
    void cacheCoolDown(size_t need_size);  // no-op in compute_new
    void cacheEvict(size_t need_size);     // no-op in compute_new
    void* cacheInsert(GlobalAddr gaddr, bool read = true);
    void markDirty(GlobalAddr gaddr);
    bool evict(MetaData* metadata);  // kept for compatibility; delegates to hot-leaf evict
    void cacheDirectEvict(size_t need_size);
    void releaseCurrentThread();
    void reset_epoch_participants();

    DataManager(RDMAClient* rdma) : rdma(rdma) {
        // Initialize the data manager
        chunk_addr.store(rdma->mem().addr);
        cache_size.store(0);

        char* limit_str = getenv("COMPUTE_CACHE_LIMIT");
        if (limit_str) {
            compute_cache_limit = strtol(limit_str, NULL, 10);
        } else {
            compute_cache_limit = 1024 * 1024 * 1024;  // 1 GiB
        }
        compute_cache_limit = (compute_cache_limit / 64) * 64; // align to 64 bytes
        // char* page_size_str = getenv("PTH_BM_PAGE_SIZE");
        // if (page_size_str) {
        //     compute_page_size = strtol(page_size_str, NULL, 10);
        // } else {
        //     compute_page_size = 4096;  // 4 KiB
        // }
        for (int i = 0; i < MAX_THREADS; i++) {
            cooling_reclaim_head[i * STRIDE] = 0;
            cooling_addrs[i].reserve(256);
            last_cxl_access[i * STRIDE].store(nullptr, std::memory_order_relaxed);
        }
        if (char* frequency_str = getenv("PTH_COLD_CACHE_FREQUENCY_NS")) {
            cold_cache_frequency_ns = strtoull(frequency_str, NULL, 10);
        }
        if (char* threshold_str = getenv("PTH_CACHE_ADMIT_THRESHOLD")) {
            cache_admit_threshold = strtoull(threshold_str, NULL, 10);
        }
        // cache_pool = aligned_alloc(64, compute_cache_limit);
        cache_pool_offset.store(0);
        std::cout << "Pentathlon compute cache limit: " << compute_cache_limit << " bytes"
                  << std::endl;
    }

    ~DataManager() {
        std::cout << "malloc size: " << malloc_size << " bytes, count: " << malloc_count
                  << std::endl;
        std::cout << "cache pool offset: " << cache_pool_offset.load() << " bytes" << std::endl;
        // for (int i = 0; i < MAX_THREADS; ++i) {
        //     std::cout << "thread " << i << " leaf lru size: " << owner_lru.size(i) << std::endl;
        // }
    }
};

#endif
