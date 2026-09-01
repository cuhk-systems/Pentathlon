#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <random>
#include <lock.h>
#include <cstdlib>
#include <cstring>

static constexpr int num_entry_in_cooling_bucket = 7;

struct CoolingBucket {
    Lock32 mutex;
    int count;
    uint64_t entries[num_entry_in_cooling_bucket];
};

class CoolingMap {
  public:
    uint64_t bucket_num_;
    uint64_t entry_num_;
    std::atomic<size_t> size_{0};
    CoolingBucket* table_;
    CoolingMap(uint64_t entry_num) {
        entry_num_ = entry_num;
        bucket_num_ = (entry_num + num_entry_in_cooling_bucket - 1) / num_entry_in_cooling_bucket;
        table_ = (CoolingBucket*) aligned_alloc(64, bucket_num_ * sizeof(CoolingBucket));
        memset(table_, 0, bucket_num_ * sizeof(CoolingBucket));
    }

    void clean_bucket(CoolingBucket* bucket) {
        for (uint32_t i = 0; i < num_entry_in_cooling_bucket; i++) {
            bucket->entries[i] = 0;
        }
    }

    void reset() {
        for (uint64_t i = 0; i < bucket_num_; i++) {
            clean_bucket(&table_[i]);
        }
        memset(table_, 0, bucket_num_ * sizeof(CoolingBucket));
    }

    size_t hash(uint64_t key) {
        return std::hash<uint64_t>{}(key) % bucket_num_;
    }

    ~CoolingMap() {
        for (uint64_t i = 0; i < bucket_num_; i++) {
            clean_bucket(&table_[i]);
        }
        free(table_);
    }

    void erase(uint64_t key) {
        uint64_t idx = hash(key);
        CoolingBucket* head = &table_[idx];
        head->mutex.get_lock();
        for (uint32_t i = 0; i < num_entry_in_cooling_bucket; i++) {
            if (head->entries[i] == key) {
                head->entries[i] = 0;
                size_.fetch_sub(1, std::memory_order_relaxed);
                for (uint32_t j = i + 1; j < num_entry_in_cooling_bucket; j++) {
                    head->entries[j - 1] = head->entries[j];
                }
                head->entries[num_entry_in_cooling_bucket - 1] = 0;
                break;
            }
        }
        head->mutex.release_lock();
    }

    inline bool empty() {
        return size_.load(std::memory_order_relaxed) == 0;
    }

    inline size_t size() {
        return size_.load(std::memory_order_relaxed);
    }

    inline uint64_t random_bucket_idx() {
        static thread_local std::mt19937 *generator = nullptr;
        if (!generator)
            generator = new std::mt19937(clock() + pthread_self());
        static thread_local std::uniform_int_distribution<uint64_t> dist(0, bucket_num_ - 1);
        return dist(*generator);
    }
};
