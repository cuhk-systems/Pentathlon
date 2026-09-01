#include "lru.hpp"

#include <atomic>

namespace {
inline LruNode* get_lru_node(MetaData* data) {
    return data ? data->lru_node : nullptr;
}
}  // namespace

void OwnerLruManager::remove(ThreadLeafLRU& list, LruNode* node) {
    if (!node) {
        return;
    }

    auto* prev = node->prev;
    auto* next = node->next;
    if (prev) {
        prev->next = next;
    } else if (list.head == node) {
        list.head = next;
    }
    if (next) {
        next->prev = prev;
    } else if (list.tail == node) {
        list.tail = prev;
    }
    if (list.size > 0) {
        --list.size;
    }
    node->prev = nullptr;
    node->next = nullptr;
}

void OwnerLruManager::release_thread(int tid) {
    if (tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return;
    }
    auto& list = leaf_lru[tid];
    auto* node = list.head;
    while (node) {
        auto* next_node = node->next;
        if (node->metadata) {
            node->metadata->owner_tid.store(-1, std::memory_order_release);
            node->in_owner_lru.store(false, std::memory_order_release);
        }
        node->prev = nullptr;
        node->next = nullptr;
        node = next_node;
    }
    list.head = nullptr;
    list.tail = nullptr;
    list.size = 0;
}

void OwnerLruManager::push_back(ThreadLeafLRU& list, LruNode* node) {
    if (!node) {
        return;
    }

    node->prev = list.tail;
    node->next = nullptr;
    if (list.tail) {
        list.tail->next = node;
    } else {
        list.head = node;
    }
    list.tail = node;
    ++list.size;
}

void OwnerLruManager::push_front(ThreadLeafLRU& list, LruNode* node) {
    if (!node) {
        return;
    }

    node->prev = nullptr;
    node->next = list.head;
    if (list.head) {
        list.head->prev = node;
    } else {
        list.tail = node;
    }
    list.head = node;
    ++list.size;
}

void OwnerLruManager::offer_front(MetaData* data, int tid, int hot_state) {
    if (!data || tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return;
    }
    auto* node = get_lru_node(data);
    if (!node) {
        return;
    }
    if (data->cache_children_num_in_state(hot_state) > 0) {
        return;
    }
    if (data->get_state() != hot_state) {
        return;
    }

    int owner = data->owner_tid.load(std::memory_order_acquire);
    if (owner == tid) {
        auto& list = leaf_lru[tid];
        if (data->owner_tid.load(std::memory_order_acquire) == tid) {
            if (node->in_owner_lru.load(std::memory_order_relaxed)) {
                remove(list, node);
            }
            push_front(list, node);
            node->in_owner_lru.store(true, std::memory_order_release);
        }
        return;
    }
    if (owner != -1) {
        return;
    }

    if (!data->owner_tid.compare_exchange_strong(owner, tid, std::memory_order_acq_rel)) {
        return;
    }

    auto& list = leaf_lru[tid];
    if (data->owner_tid.load(std::memory_order_acquire) == tid) {
        if (node->in_owner_lru.load(std::memory_order_relaxed)) {
            remove(list, node);
        }
        push_front(list, node);
        node->in_owner_lru.store(true, std::memory_order_release);
    }
}

void OwnerLruManager::offer(MetaData* data, int tid, int hot_state) {
    if (!data || tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return;
    }
    auto* node = get_lru_node(data);
    if (!node) {
        return;
    }
    if (data->cache_children_num_in_state(hot_state) > 0) {
        return;
    }
    if (data->get_state() != hot_state) {
        return;
    }

    int owner = data->owner_tid.load(std::memory_order_acquire);
    if (owner == tid) {
        auto& list = leaf_lru[tid];
        if (data->owner_tid.load(std::memory_order_acquire) == tid) {
            if (node->in_owner_lru.load(std::memory_order_relaxed)) {
                remove(list, node);
            }
            push_back(list, node);
            node->in_owner_lru.store(true, std::memory_order_release);
        }
        return;
    }
    if (owner != -1) {
        return;
    }

    if (!data->owner_tid.compare_exchange_strong(owner, tid, std::memory_order_acq_rel)) {
        return;
    }

    auto& list = leaf_lru[tid];
    if (data->owner_tid.load(std::memory_order_acquire) == tid) {
        if (node->in_owner_lru.load(std::memory_order_relaxed)) {
            remove(list, node);
        }
        push_back(list, node);
        node->in_owner_lru.store(true, std::memory_order_release);
    }
}

void OwnerLruManager::touch_if_owned(MetaData* data, int tid) {
    if (!data || tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return;
    }
    auto* node = get_lru_node(data);
    if (!node) {
        return;
    }
    if (data->owner_tid.load(std::memory_order_acquire) != tid ||
        !node->in_owner_lru.load(std::memory_order_acquire)) {
        return;
    }

    auto& list = leaf_lru[tid];
    if (data->owner_tid.load(std::memory_order_acquire) == tid &&
        node->in_owner_lru.load(std::memory_order_relaxed)) {
        remove(list, node);
        push_back(list, node);
        node->in_owner_lru.store(true, std::memory_order_release);
    }
}

void OwnerLruManager::remove_if_owned(MetaData* data, int tid) {
    if (!data || tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return;
    }
    auto* node = get_lru_node(data);
    if (!node) {
        return;
    }

    int owner = data->owner_tid.load(std::memory_order_acquire);
    if (owner != tid || !node->in_owner_lru.load(std::memory_order_acquire)) {
        return;
    }

    auto& list = leaf_lru[tid];
    if (data->owner_tid.load(std::memory_order_acquire) == tid &&
        node->in_owner_lru.load(std::memory_order_relaxed)) {
        remove(list, node);
        node->in_owner_lru.store(false, std::memory_order_release);
        data->owner_tid.store(-1, std::memory_order_release);
    }
}

MetaData* OwnerLruManager::pop_head_candidate(int tid) {
    if (tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return nullptr;
    }

    auto& list = leaf_lru[tid];
    auto* node = list.head;
    if (!node) {
        return nullptr;
    }

    remove(list, node);
    node->in_owner_lru.store(false, std::memory_order_release);
    return node->metadata;
}

void OwnerLruManager::release_owner(MetaData* data) {
    auto* node = get_lru_node(data);
    if (!data || !node) {
        return;
    }

    int owner = data->owner_tid.load(std::memory_order_acquire);
    if (owner >= 0 && owner < static_cast<int>(kComputeMaxThreads) &&
        node->in_owner_lru.load(std::memory_order_acquire)) {
        remove(leaf_lru[owner], node);
    }
    node->in_owner_lru.store(false, std::memory_order_release);
    data->owner_tid.store(-1, std::memory_order_release);
    node->prev = nullptr;
    node->next = nullptr;
}

size_t OwnerLruManager::size(int tid) const {
    if (tid < 0 || tid >= static_cast<int>(kComputeMaxThreads)) {
        return 0;
    }
    return leaf_lru[tid].size;
}
