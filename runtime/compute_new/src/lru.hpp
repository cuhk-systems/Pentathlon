#ifndef COMPUTE_LRU_HPP
#define COMPUTE_LRU_HPP

#include <atomic>
#include <cstddef>

#include "metadata.hpp"

constexpr size_t kComputeMaxThreads = 64;

struct LruNode {
    MetaData* metadata;
    std::atomic<bool> in_owner_lru;
    LruNode* prev;
    LruNode* next;

    explicit LruNode(MetaData* metadata_ptr = nullptr)
        : metadata(metadata_ptr), in_owner_lru(false), prev(nullptr), next(nullptr) {}
};

struct alignas(128) ThreadLeafLRU {
    LruNode* head;
    LruNode* tail;
    size_t size;

    ThreadLeafLRU() : head(nullptr), tail(nullptr), size(0) {}
};

class OwnerLruManager {
   public:
    void offer(MetaData* data, int tid, int hot_state);
    void offer_front(MetaData* data, int tid, int hot_state);
    void touch_if_owned(MetaData* data, int tid);
    void remove_if_owned(MetaData* data, int tid);
    MetaData* pop_head_candidate(int tid);
    void release_owner(MetaData* data);
    size_t size(int tid) const;
    void release_thread(int tid);

   private:
    ThreadLeafLRU leaf_lru[kComputeMaxThreads];

    void remove(ThreadLeafLRU& list, LruNode* node);
    void push_back(ThreadLeafLRU& list, LruNode* node);
    void push_front(ThreadLeafLRU& list, LruNode* node);
};

#endif
