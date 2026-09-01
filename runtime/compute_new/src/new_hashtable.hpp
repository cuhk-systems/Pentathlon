#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <random>
#include <lock.h>
#include <cstdlib>
#include <cstring>

static constexpr int num_entry_in_bucket = 3;

struct frame_t {
    uint64_t key, value;
};

struct bucket_t {
    Lock32 mutex;
    uint32_t count;
    bucket_t* next;
    frame_t entries[num_entry_in_bucket];
};

class Hashtable {
  public:
    uint64_t bucket_num_;
    uint64_t entry_num_;
    std::atomic<size_t> size_{0};
    bucket_t* table_;
    Hashtable(uint64_t entry_num) {
        entry_num_ = entry_num;
        bucket_num_ = (entry_num + num_entry_in_bucket - 1) / num_entry_in_bucket;
        table_ = (bucket_t*) aligned_alloc(64, bucket_num_ * sizeof(bucket_t));
    }

    void clean_bucket(bucket_t* bucket) {
        if (bucket == nullptr) return;
        clean_bucket(bucket->next);
        free(bucket);
    }

    void reset() {
        for (uint64_t i = 0; i < bucket_num_; i++) {
            clean_bucket(table_[i].next);
        }
        memset(table_, 0, bucket_num_ * sizeof(bucket_t));
    }

    size_t hash(uint64_t key) {
        return std::hash<uint64_t>{}(key + 19260817) % bucket_num_;
    }

    ~Hashtable() {
        for (uint64_t i = 0; i < bucket_num_; i++) {
            clean_bucket(table_[i].next);
        }
        free(table_);
    }

    void insert(uint64_t key, uint64_t value) {
        uint64_t idx = hash(key);
        bucket_t* head = &table_[idx];
        head->mutex.get_lock();
        bucket_t* bucket = head;
        bucket_t* prev = nullptr;
        while (bucket != nullptr) {
            for (uint32_t i = 0; i < num_entry_in_bucket; i++) {
                if (bucket->entries[i].key == key) {
                    bucket->entries[i].value = value;
                    head->mutex.release_lock();
                    return;
                }
                if (bucket->entries[i].key == 0) {
                    bucket->entries[i] = {key, value};
                    bucket->count++;
                    size_.fetch_add(1, std::memory_order_relaxed);
                    head->mutex.release_lock();
                    return;
                }
            }
            prev = bucket;
            bucket = bucket->next;
        }
        bucket_t* new_bucket = (bucket_t*) malloc(sizeof(bucket_t));
        memset(new_bucket, 0, sizeof(bucket_t));
        new_bucket->entries[0] = {key, value};
        new_bucket->count = 1;
        size_.fetch_add(1, std::memory_order_relaxed);
        prev->next = new_bucket;
        head->mutex.release_lock();
    }

    void erase(uint64_t key) {
        uint64_t idx = hash(key);
        bucket_t* head = &table_[idx];
        head->mutex.get_lock();
        bucket_t* bucket = head;
        bucket_t* prev = nullptr;
        while (bucket != nullptr) {
            for (uint32_t i = 0; i < num_entry_in_bucket; i++) {
                if (bucket->entries[i].key == key) {
                    bucket->count--;
                    bucket->entries[i] = {0, 0};
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    if (bucket->count == 0 && prev != nullptr) {
                        prev->next = bucket->next;
                        free(bucket);
                    }
                    head->mutex.release_lock();
                    return;
                }
            }
            prev = bucket;
            bucket = bucket->next;
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
