#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include "addr.h"
#include "data-manager.hpp"
#include "init.hpp"

namespace {
constexpr uint8_t kStateInCacheHot = 3;
constexpr uint8_t kStateInCacheCooling = 2;
constexpr int kBackoffSpinCount = 32;
constexpr int kBackoffYieldCount = 96;
constexpr int kBackoffSleepNs = 50;

class ContentionBackoff {
   public:
    ContentionBackoff() : attempts_(0) {}

    void wait() {
        if (attempts_ < kBackoffSpinCount) {
            for (int i = 0; i < 32; ++i) {
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        } else if (attempts_ < kBackoffYieldCount) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::nanoseconds(kBackoffSleepNs));
        }
        ++attempts_;
    }

   private:
    int attempts_;
};

inline MetaDataAddr* get_valid_metadata_addr(MetaData* owner) {
    if (!owner || !owner->local_addr) {
        return nullptr;
    }
    auto* metadata_addr = metadata_addr_from_local_addr(owner->local_addr);
    if (reinterpret_cast<MetaData*>(metadata_addr->get_addr()) != owner) {
        return nullptr;
    }
    if (metadata_addr->state.load(std::memory_order_relaxed) != kStateInCacheHot) {
        return nullptr;
    }
    return metadata_addr;
}

inline int cache_children_num_load(MetaData* metadata) {
    if (!metadata) {
        return 0;
    }
    auto* metadata_addr = get_valid_metadata_addr(metadata);
    if (metadata_addr) {
        return metadata_addr->cache_children_num.load(std::memory_order_relaxed);
    }
    return 0;
}

inline int cache_children_num_fetch_add(MetaData* metadata, int delta) {
    auto* metadata_addr = get_valid_metadata_addr(metadata);
    if (metadata_addr) {
        return metadata_addr->cache_children_num.fetch_add(delta, std::memory_order_relaxed);
    }
    return 0;
}

inline int cache_children_num_fetch_sub(MetaData* metadata, int delta) {
    auto* metadata_addr = get_valid_metadata_addr(metadata);
    if (metadata_addr) {
        return metadata_addr->cache_children_num.fetch_sub(delta, std::memory_order_relaxed);
    }
    return 0;
}

inline bool try_update_parent_ptr(MetaData* child, void* child_local_addr, MetaData* parent,
                                  int parent_offset, bool to_local) {
    if (!child || !child_local_addr || !parent) {
        return false;
    }

    if (to_local) {
        if (!get_valid_metadata_addr(parent)) {
            return false;
        }
        void* ptr_raw = static_cast<char*>(parent->local_addr) + parent_offset;
        GlobalAddr old_addr =
            GlobalAddr::fromPointer(__atomic_load_n((void**)ptr_raw, __ATOMIC_ACQUIRE));
        if (!old_addr.global_tag) {
            return false;
        }
        if (old_addr.local_tag || reinterpret_cast<MetaData*>(old_addr.addr) != child) {
            return false;
        }
        auto new_addr = GlobalAddr{old_addr.offset, reinterpret_cast<uint64_t>(child_local_addr), 1, 1};
        void* old_addr_val = (void*)old_addr.val;
        void* new_addr_val = (void*)new_addr.val;
        // printf("================== Attempting to update parent pointer from %p to %p for local cache insertion =================\n", old_addr_val, new_addr_val);
        // printf("Parent metadata: %p, child metadata: %p\n", parent, child);
        // printf("Parent offset: %d, pointer address: %p\n", parent_offset, ptr_raw);
        // printf("Old address details - offset: %u, addr: %p, local_tag: %u, global_tag: %u\n",
        //        old_addr.offset, reinterpret_cast<void*>(old_addr.addr), old_addr.local_tag, old_addr.global_tag);
        // printf("New address details - offset: %u, addr: %p, local_tag: %u, global_tag: %u\n",
        //        new_addr.offset, reinterpret_cast<void*>(new_addr.addr), new_addr.local_tag, new_addr.global_tag);
        // printf("Child local address: %p\n", child_local_addr);
        return __atomic_compare_exchange_n((void**)ptr_raw, &old_addr_val, new_addr_val, false,
                                           __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }

    auto unswizzle_at = [&](void* ptr_raw) {
        if (!ptr_raw) {
            return false;
        }
        GlobalAddr old_addr =
            GlobalAddr::fromPointer(__atomic_load_n((void**)ptr_raw, __ATOMIC_ACQUIRE));
        if (!old_addr.global_tag || !old_addr.local_tag ||
            reinterpret_cast<void*>(old_addr.addr) != child_local_addr) {
            return false;
        }
        auto new_addr =
            GlobalAddr{old_addr.offset, reinterpret_cast<uint64_t>(child), 0, 1};
        void* old_addr_val = reinterpret_cast<void*>(old_addr.val);
        void* new_addr_val = reinterpret_cast<void*>(new_addr.val);
        return __atomic_compare_exchange_n((void**)ptr_raw, &old_addr_val, new_addr_val, false,
                                           __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    };

    bool local_updated = false;
    void* parent_local_addr = parent->local_addr;
    if (parent_local_addr) {
        auto* header = metadata_addr_from_local_addr(parent_local_addr);
        auto state = header->state.load(std::memory_order_acquire);
        if (reinterpret_cast<MetaData*>(header->get_addr()) == parent &&
            (state == kStateInCacheHot || state == kStateInCacheCooling)) {
            local_updated =
                unswizzle_at(static_cast<char*>(parent_local_addr) + parent_offset);
        }
    }
    bool remote_updated =
        unswizzle_at(static_cast<char*>(parent->remote_addr) + parent_offset);
    if (local_updated) {
        parent->dirty.store(true, std::memory_order_relaxed);
    }
    return local_updated || remote_updated;
}

}  // namespace

void DataManager::ensure_epoch_registered(int tid) {
    if (tid < 0 || tid >= static_cast<int>(MAX_THREADS)) {
        return;
    }
    auto& slot = epoch_states[tid * STRIDE];
    if (!slot.active.load(std::memory_order_acquire)) {
        slot.epoch.store(global_epoch.load(std::memory_order_acquire), std::memory_order_release);
        slot.active.store(true, std::memory_order_release);
    }
}

void DataManager::publish_quiescent_epoch(int tid) {
    if (tid < 0 || tid >= static_cast<int>(MAX_THREADS)) {
        return;
    }
    ensure_epoch_registered(tid);
    epoch_states[tid * STRIDE].epoch.store(global_epoch.load(std::memory_order_acquire),
                                           std::memory_order_release);
}

void DataManager::on_node_became_hot(MetaData* metadata, MetaDataAddr* metadata_addr, int tid) {
    auto parent = metadata->father.load(std::memory_order_relaxed);
    if (parent != nullptr) {
        if (cache_children_num_fetch_add(parent, 1) == 0) {
            owner_lru.remove_if_owned(parent, tid);
        }
        auto father_offset = metadata->father_offset.load(std::memory_order_relaxed);
        try_update_parent_ptr(metadata, metadata->local_addr, parent, father_offset, true);
    }
    if (metadata_addr->cache_children_num.load(std::memory_order_relaxed) == 0) {
        owner_lru.offer(metadata, tid, state_in_cache_hot);
    }
}

bool DataManager::can_reclaim_epoch(uint64_t retire_epoch) {
    for (size_t i = 0; i < MAX_THREADS; ++i) {
        auto& slot = epoch_states[i * STRIDE];
        if (!slot.active.load(std::memory_order_acquire)) {
            continue;
        }
        uint64_t thread_epoch = slot.epoch.load(std::memory_order_acquire);
        if (thread_epoch <= retire_epoch) {
            return false;
        }
    }
    return true;
}

void DataManager::retire_cooling_addr(int tid, MetaData* metadata, void* local_addr, size_t size) {
    if (tid < 0 || tid >= static_cast<int>(MAX_THREADS)) {
        return;
    }
    uint64_t retire_epoch = global_epoch.fetch_add(1, std::memory_order_acq_rel);
    if (metadata) {
        metadata->cooling_epoch.store(retire_epoch, std::memory_order_release);
    }
    cooling_addrs[tid].push_back({metadata, local_addr, size, retire_epoch});
}

void DataManager::reclaim_cooling_addrs(int tid) {
    if (tid < 0 || tid >= static_cast<int>(MAX_THREADS)) {
        return;
    }

    auto& retired = cooling_addrs[tid];
    auto& head = cooling_reclaim_head[tid * STRIDE];
    if (head >= retired.size()) {
        if (!retired.empty()) {
            retired.clear();
        }
        head = 0;
        return;
    }

    size_t stop = retired.size();
    while (head < stop) {
        auto entry = retired[head];
        if (!can_reclaim_epoch(entry.retire_epoch)) {
            break;
        }
        bool reclaimed = false;
        bool current_retire = entry.metadata &&
                              entry.metadata->cooling_epoch.load(std::memory_order_acquire) ==
                                  entry.retire_epoch;
        if (entry.metadata && entry.local_addr &&
            entry.metadata->local_addr == entry.local_addr && current_retire) {
            auto* metadata_addr = metadata_addr_from_local_addr(entry.local_addr);
            if (reinterpret_cast<MetaData*>(metadata_addr->get_addr()) == entry.metadata) {
                uint8_t expected_state = state_in_cache_cooling;
                if (metadata_addr->state.compare_exchange_strong(
                        expected_state, state_in_cache_reclaiming, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    auto* parent = entry.metadata->father.load(std::memory_order_relaxed);
                    if (parent) {
                        auto parent_offset =
                            entry.metadata->father_offset.load(std::memory_order_relaxed);
                        try_update_parent_ptr(entry.metadata, entry.local_addr, parent,
                                              parent_offset, false);
                    }
                    if (entry.metadata->dirty.load(std::memory_order_relaxed)) {
                        rdma->write(entry.local_addr, entry.metadata->remote_addr,
                                    entry.metadata->size);
                        entry.metadata->dirty.store(false, std::memory_order_relaxed);
                    }
                    entry.metadata->access_count.store(0, std::memory_order_relaxed);
                    metadata_addr->state.store(state_not_in_cache, std::memory_order_release);
                    entry.metadata->local_addr = nullptr;
                    evicted_addrs[tid].push_back(entry.local_addr);
                    reclaimed = true;
                }
            }
        }

        if (!reclaimed && current_retire && entry.metadata && entry.local_addr &&
            entry.metadata->local_addr == entry.local_addr &&
            entry.metadata->get_state() == state_in_cache_cooling) {
            retired.push_back(entry);
        }

        ++head;
    }

    if (head > 1024 && head * 2 >= retired.size()) {
        retired.erase(retired.begin(), retired.begin() + static_cast<std::ptrdiff_t>(head));
        head = 0;
    }
}

void DataManager::reset_epoch_participants() {
    uint64_t epoch = global_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    for (size_t i = 0; i < MAX_THREADS; ++i) {
        last_cxl_access[i * STRIDE].store(nullptr, std::memory_order_seq_cst);
        auto& slot = epoch_states[i * STRIDE];
        slot.epoch.store(epoch, std::memory_order_release);
        slot.active.store(false, std::memory_order_release);
    }
}

bool DataManager::try_evict_hot_leaf(MetaData* metadata) {
    if (!metadata) {
        return false;
    }
    auto local_addr = metadata->local_addr;
    if (!local_addr) {
        return false;
    }
    if (cache_children_num_load(metadata) > 0) {
        return false;
    }

    auto metadata_addr = metadata_addr_from_local_addr(local_addr);

    auto parent = metadata->father.load(std::memory_order_relaxed);
    if (metadata->cache_children_num_in_state(state_in_cache_hot) > 0) {
        return false;
    }

    uint8_t expected_state = state_in_cache_hot;
    if (!metadata_addr->state.compare_exchange_strong(expected_state, state_in_cache_cooling,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
        return false;
    }

    if (parent) {
        // Detach the node as part of hot -> cooling. Threads that already
        // observed it remain protected by the retire epoch.
        auto parent_offset = metadata->father_offset.load(std::memory_order_relaxed);
        try_update_parent_ptr(metadata, local_addr, parent, parent_offset, false);
        if (cache_children_num_fetch_sub(parent, 1) == 1 &&
            parent->get_state() == state_in_cache_hot) {
            owner_lru.offer(parent, get_thread_id(), state_in_cache_hot);
        }
    }

    // metadata->father.store(nullptr, std::memory_order_relaxed);
    // metadata->father_offset.store(0, std::memory_order_relaxed);

    owner_lru.release_owner(metadata);

    int id = get_thread_id();
    retire_cooling_addr(id, metadata, local_addr,
                        static_cast<size_t>((metadata->size + sizeof(MetaDataAddr) + 63) & ~63));
    // free(local_addr);
    return true;
}

void DataManager::disaggFree(GlobalAddr gaddr) {
    if (!gaddr.global_tag) {
        fprintf(stderr, "Error: Invalid address\n");
        return;
    }

    MetaData* metadata = nullptr;
    if (gaddr.local_tag) {
        auto metadata_addr = metadata_addr_from_local_addr(reinterpret_cast<void*>(gaddr.addr));
        metadata = reinterpret_cast<MetaData*>(metadata_addr->get_addr());
    } else {
        metadata = reinterpret_cast<MetaData*>(gaddr.addr);
    }
    if (metadata == nullptr) {
        fprintf(stderr, "Error: Address not found in address table\n");
        return;
    }

    owner_lru.release_owner(metadata);
    metadata->local_addr = nullptr;
    delete metadata->lru_node;
    delete metadata;
}

GlobalAddr DataManager::disaggAlloc(size_t size) {
    GlobalAddr addr = GlobalAddr::null();
    assert(size <= MAX_SIZE);

    struct ThreadChunkCursors {
        const DataManager* owner = nullptr;
        std::array<uint64_t, MAX_SIZE + 1> next{};
    };
    static thread_local ThreadChunkCursors cursors;

    if (cursors.owner != this) {
        cursors.next.fill(0);
        cursors.owner = this;
    }

    const size_t page_size = chunk_size();
    uint64_t& next_addr = cursors.next[size];
    if (next_addr == 0 || (next_addr & (page_size - 1)) + size >= page_size) {
        next_addr = chunk_addr.fetch_add(page_size, std::memory_order_relaxed);
    }
    uint64_t remote_addr = next_addr;
    next_addr += size;

    auto metadata = new (std::align_val_t(64)) MetaData();
    if (metadata == nullptr) {
        throw std::runtime_error("out of memory in compute side");
    }
    metadata->lru_node = new LruNode(metadata);
    if (metadata->lru_node == nullptr) {
        delete metadata;
        throw std::runtime_error("out of memory allocating LRU node");
    }

    addr.global_tag = 1;
    addr.local_tag = 0;
    addr.offset = 0;
    addr.addr = reinterpret_cast<uint64_t>(metadata);

    metadata->size = static_cast<int>(size);
    metadata->remote_addr = reinterpret_cast<void*>(remote_addr);
    metadata->local_addr = nullptr;
    metadata->dirty.store(false, std::memory_order_relaxed);

    malloc_size += size;
    malloc_count++;
    cacheInsert(addr, false);
    return addr;
}

void DataManager::addAddrDep(GlobalAddr addr_u, GlobalAddr addr_v) {
    if (!addr_u.global_tag || !addr_v.global_tag) {
        return;
    }
    if (addr_v.local_tag) {
        return;
    }

    MetaData *metadata_u, *metadata_v;
    // cacheInsert(addr_v, true);
    if (addr_u.local_tag) {
        auto metadata_u_addr = metadata_addr_from_local_addr(reinterpret_cast<void*>(addr_u.addr));
        metadata_u = reinterpret_cast<MetaData*>(metadata_u_addr->get_addr());
    } else {
        metadata_u = reinterpret_cast<MetaData*>(addr_u.addr);
    }
    metadata_v = reinterpret_cast<MetaData*>(addr_v.addr);

    if (metadata_u == nullptr || metadata_v == nullptr || metadata_u == metadata_v) {
        return;
    }
    if (metadata_u->father.load() == metadata_v || metadata_v->father.load() == metadata_u) {
        return;
    }

    auto old_father = metadata_v->father.load(std::memory_order_relaxed);
    auto old_father_offset = metadata_v->father_offset.load(std::memory_order_relaxed);
    metadata_v->father.store(metadata_u, std::memory_order_relaxed);
    metadata_v->father_offset.store(addr_u.offset, std::memory_order_relaxed);

    if (metadata_v->get_state() == state_in_cache_hot) {
        if (old_father != nullptr) {
            try_update_parent_ptr(metadata_v, metadata_v->local_addr, old_father,
                                  old_father_offset, false);
            if (cache_children_num_fetch_sub(old_father, 1) == 1 &&
                old_father->get_state() == state_in_cache_hot) {
                owner_lru.offer(old_father, get_thread_id(), state_in_cache_hot);
            }
        }

        if (cache_children_num_fetch_add(metadata_u, 1) == 0) {
            owner_lru.remove_if_owned(metadata_u, get_thread_id());
        }
    }
}

// GlobalAddr DataManager::relAddrDep(GlobalAddr ptr) {
//     // printf("relAddrDep called with gaddr {offset: %u, addr: %p, local_tag: %u, global_tag: %u}\n",
//     //        ptr.offset, reinterpret_cast<void*>(ptr.addr), ptr.local_tag, ptr.global_tag);
//     if (!ptr.global_tag || !ptr.local_tag) {
//         return ptr;
//     }
//     auto metadata_addr = metadata_addr_from_local_addr(reinterpret_cast<void*>(ptr.addr));
//     auto metadata = reinterpret_cast<MetaData*>(metadata_addr->get_addr());
//     auto old_father = metadata->father.load(std::memory_order_relaxed);
//     auto old_father_offset = metadata->father_offset.load(std::memory_order_relaxed);
//     try_update_parent_ptr(metadata, reinterpret_cast<void*>(ptr.addr), old_father, old_father_offset, false);
//     if (metadata == nullptr) {
//         throw std::runtime_error("metadata not found in relAddrDep");
//     }
//     auto new_gaddr = GlobalAddr{ptr.offset, reinterpret_cast<uint64_t>(metadata), 0, 1};
//     if (ptr.offset > metadata->size) {
//         throw std::runtime_error("offset exceeds metadata size in relAddrDep");
//     }
//     // print debug info
//     // printf("relAddrDep: local addr = %p, metadata = %p, new_gaddr = {offset: %u, addr: %p, local_tag: %u, global_tag: %u}\n",
//     //        reinterpret_cast<void*>(ptr.addr), metadata, new_gaddr.offset, reinterpret_cast<void*>(new_gaddr.addr),
//     //        new_gaddr.local_tag, new_gaddr.global_tag);
//     if (metadata != reinterpret_cast<MetaData*>(new_gaddr.addr)) {
//         printf("Metadata mismatch in relAddrDep: expected %p, got %p\n", metadata, reinterpret_cast<void*>(new_gaddr.addr));
//         throw std::runtime_error("metadata mismatch in relAddrDep");
//     }
//     return new_gaddr;
// }

void* DataManager::updateAddrDep(GlobalAddr father, GlobalAddr child) {
    if (!child.global_tag || !child.local_tag || !father.global_tag) {
        return reinterpret_cast<void*>(child.val);
    }
    // update father and father_offset in child's metadata
    MetaDataAddr* metadata_addr = metadata_addr_from_local_addr(reinterpret_cast<void*>(child.addr));
    MetaData* metadata = reinterpret_cast<MetaData*>(metadata_addr->get_addr());
    if (metadata == nullptr) {
        throw std::runtime_error("metadata not found in updateAddrDep");
    }
    if (!father.global_tag) {
        GlobalAddr new_gaddr = GlobalAddr{child.offset, reinterpret_cast<uint64_t>(metadata), 0, 1};
        return reinterpret_cast<void*>(new_gaddr.val);
    }
    // release old father if exists
    auto old_father = metadata->father.load(std::memory_order_relaxed);
    if (old_father) {
        auto old_father_offset = metadata->father_offset.load(std::memory_order_relaxed);
        try_update_parent_ptr(metadata, reinterpret_cast<void*>(child.addr), old_father,
                              old_father_offset, false);
        if (cache_children_num_fetch_sub(old_father, 1) == 1 &&
            old_father->get_state() == state_in_cache_hot) {
            owner_lru.offer(old_father, get_thread_id(), state_in_cache_hot);
            // owner_lru.offer_front(old_father, get_thread_id(), state_in_cache_hot);
        }
    }
    MetaData* father_metadata = nullptr;
    MetaDataAddr* father_metadata_addr = nullptr;
    if (father.local_tag) {
        father_metadata_addr = metadata_addr_from_local_addr(reinterpret_cast<void*>(father.addr));
        father_metadata = reinterpret_cast<MetaData*>(father_metadata_addr->get_addr());
    } else {
        father_metadata = reinterpret_cast<MetaData*>(father.addr);
    }
    metadata->father.store(father_metadata, std::memory_order_relaxed);
    metadata->father_offset.store(father.offset, std::memory_order_relaxed);
    if (metadata->get_state() == state_in_cache_hot) {
        if (cache_children_num_fetch_add(father_metadata, 1) == 0) {
            owner_lru.remove_if_owned(father_metadata, get_thread_id());
        }
    }

    // A direct-CXL parent cannot be rewritten by eager unswizzle because it has
    // no local cache image. Never persist a cache-local child pointer into it.
    if (!father.local_tag) {
        GlobalAddr stable_child = {
            child.offset, reinterpret_cast<uint64_t>(metadata), 0, 1};
        return reinterpret_cast<void*>(stable_child.val);
    }
    return reinterpret_cast<void*>(child.val);
}

void* DataManager::getLocalAddr(GlobalAddr gaddr) {
    size_t offset = gaddr.offset;
    auto local_addr = cacheInsert(gaddr);
    if (!local_addr) {
        throw std::runtime_error("failed to insert address into cache");
    }
    return (void*)((char*)local_addr + offset);
}

void* DataManager::cacheInsert(GlobalAddr gaddr, bool read) {
    if (!gaddr.global_tag) {
        throw std::runtime_error("not a global address");
    }

    int tid = get_thread_id();
    auto& current_cxl_access = last_cxl_access[tid * STRIDE];
    current_cxl_access.store(nullptr, std::memory_order_seq_cst);
    ensure_epoch_registered(tid);
    auto finish_cache_insert = [](void* addr) -> void* { return addr; };
    MetaData* metadata = nullptr;
    // printf("Thread %d: cacheInsert called with gaddr {offset: %u, addr: %p, local_tag: %u, global_tag: %u}\n",
    //        tid, gaddr.offset, reinterpret_cast<void*>(gaddr.addr), gaddr.local_tag, gaddr.global_tag);

    if (gaddr.local_tag) {
        return finish_cache_insert(reinterpret_cast<void*>(gaddr.addr));
    } else {
        metadata = reinterpret_cast<MetaData*>(gaddr.addr);
        if (metadata == nullptr) {
            throw std::runtime_error("metadata not found");
        }
    }

    auto state = metadata->get_state();
    if (state == state_in_cache_hot) {
        void* local_addr = metadata->local_addr;
        if (local_addr == nullptr) {
            throw std::runtime_error("metadata in hot state but local_addr is null");
        }
        auto parent = metadata->father.load(std::memory_order_relaxed);
        // if (parent != nullptr) {
        //     auto father_offset = metadata->father_offset.load(std::memory_order_relaxed);
        //     try_update_parent_ptr(metadata, local_addr, parent, father_offset, true);
        // }
        if (cache_children_num_load(metadata) == 0) {
            owner_lru.offer(metadata, tid, state_in_cache_hot);
        }
        return finish_cache_insert(local_addr);
    }

    if (state == state_in_cache_cooling) {
        void* local_addr = metadata->local_addr;
        if (local_addr != nullptr) {
            auto* metadata_addr = metadata_addr_from_local_addr(local_addr);
            if (reinterpret_cast<MetaData*>(metadata_addr->get_addr()) == metadata) {
                uint8_t expected_state = state_in_cache_cooling;
                if (metadata_addr->state.compare_exchange_strong(
                        expected_state, state_in_cache_hot, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    on_node_became_hot(metadata, metadata_addr, tid);
                    return finish_cache_insert(local_addr);
                }

                if (expected_state == state_in_cache_hot) {
                    return finish_cache_insert(local_addr);
                }
            }
        }
        state = metadata->get_state();
    }

    if (state == state_in_cache_reclaiming) {
        ContentionBackoff backoff;
        while (metadata->get_state() == state_in_cache_reclaiming) {
            backoff.wait();
        }
        return cacheInsert(gaddr, read);
    }

    // If the metadata is not in cache hot state
    // Publish the candidate before validating loading/state. Once loading_flag is
    // set, no new direct access can pass validation and the loader can drain slots.
    // Otherwise, try to load it into the cache and set the state to hot, do not add access count
#ifdef PENTATHLON_USE_CXL
    if (state == state_not_in_cache) {
        current_cxl_access.store(metadata, std::memory_order_seq_cst);
        if (!metadata->loading_flag.load(std::memory_order_seq_cst) &&
            metadata->get_state() == state_not_in_cache) {
            uint64_t access_count = metadata->access_count.load(std::memory_order_relaxed);
            while (access_count < cache_admit_threshold) {
                if (metadata->access_count.compare_exchange_weak(
                        access_count, access_count + 1, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    return finish_cache_insert(metadata->remote_addr);
                }
            }
        }
        current_cxl_access.store(nullptr, std::memory_order_seq_cst);
    }
#endif

    bool expected = false;
    if (metadata->get_state() != state_in_cache_hot && metadata->loading_flag.compare_exchange_strong(expected, true)) {
        if (metadata->get_state() != state_in_cache_hot) {
#ifdef PENTATHLON_USE_CXL
            ContentionBackoff direct_access_backoff;
            for (;;) {
                bool direct_access_in_progress = false;
                for (size_t i = 0; i < MAX_THREADS; ++i) {
                    if (static_cast<int>(i) == tid) {
                        continue;
                    }
                    if (last_cxl_access[i * STRIDE].load(std::memory_order_seq_cst) ==
                        metadata) {
                        direct_access_in_progress = true;
                        break;
                    }
                }
                if (!direct_access_in_progress) {
                    break;
                }
                direct_access_backoff.wait();
            }
#endif
            void* local_addr = assignLocalAddr(metadata, read);
            if (local_addr == nullptr) {
#ifdef PENTATHLON_USE_CXL
                current_cxl_access.store(metadata, std::memory_order_seq_cst);
                metadata->loading_flag.store(false, std::memory_order_seq_cst);
                return finish_cache_insert(metadata->remote_addr);
#else
                metadata->loading_flag.store(false, std::memory_order_release);
                return cacheInsert(gaddr, read);
#endif
            }
            auto metadata_addr = metadata_addr_from_local_addr(local_addr);
            metadata_addr->state.store(state_in_cache_hot, std::memory_order_release);
            on_node_became_hot(metadata, metadata_addr, tid);
        }
        metadata->loading_flag.store(false, std::memory_order_release);
    } else {
        ContentionBackoff backoff;
        while (metadata->loading_flag.load(std::memory_order_acquire)) {
            backoff.wait();
            // std::this_thread::yield();
        }
        return cacheInsert(gaddr, read);
    }

    auto* local_addr = metadata->local_addr;
    if (local_addr == nullptr) {
        return cacheInsert(gaddr, read);
    }
    return finish_cache_insert(local_addr);
}

void* DataManager::assignLocalAddr(MetaData* metadata, bool read) {
    if (metadata == nullptr) {
        throw std::runtime_error("metadata is null in assignLocalAddr");
    }
    auto size = metadata->size + sizeof(MetaDataAddr);
    size = (size + 63) & ~63;
    void* local_addr = nullptr;
    // metadata->father.store(nullptr, std::memory_order_relaxed);
    // metadata->father_offset.store(0, std::memory_order_relaxed);
    if (!assignFinished.load(std::memory_order_relaxed)) {
        size_t offset = cache_pool_offset.fetch_add(size);
        if (offset + size <= compute_cache_limit) {
            // local_addr = (void*)((char*)cache_pool + offset + sizeof(MetaDataAddr));
            auto new_addr = (void*) aligned_alloc(64, size);
            local_addr = (void*)((char*)new_addr + sizeof(MetaDataAddr));
            if (read) {
                rdma->read(metadata->remote_addr, local_addr, metadata->size);
            }
            MetaDataAddr* metadata_addr = metadata_addr_from_local_addr(local_addr);
            metadata_addr->reset();
            metadata_addr->set_addr(reinterpret_cast<uint64_t>(metadata));
            assert(metadata_addr->get_addr() == metadata);
            metadata->local_addr = local_addr;
            return local_addr;
        }
        assignFinished.store(true, std::memory_order_relaxed);
    }

    int id = get_thread_id();
    publish_quiescent_epoch(id);
    reclaim_cooling_addrs(id);
    if (evicted_addrs[id].empty()) {
        cacheDirectEvict(size);
        publish_quiescent_epoch(id);
        reclaim_cooling_addrs(id);
    }
    if (evicted_addrs[id].empty()) {
        return nullptr;
    }
    local_addr = evicted_addrs[id].back();
    evicted_addrs[id].pop_back();
    // printf("Thread %d assigned local address %p for metadata %p (size: %d)\n", id, local_addr, metadata, metadata->size);
    if (!local_addr) {
        throw std::runtime_error("out of memory in compute cache");
    }
    free((char*)local_addr - sizeof(MetaDataAddr));
    auto new_addr = (void*) aligned_alloc(64, size);
    local_addr = (void*)((char*)new_addr + sizeof(MetaDataAddr));
    if (read) {
        rdma->read(metadata->remote_addr, local_addr, metadata->size);
    }
    MetaDataAddr* metadata_addr = metadata_addr_from_local_addr(local_addr);
    metadata_addr->reset();
    metadata_addr->set_addr(reinterpret_cast<uint64_t>(metadata));
    assert(metadata_addr->get_addr() == metadata);
    // if (evicted_addrs[id].empty()) {
    //     cacheDirectEvict(size);
    // }
    metadata->local_addr = local_addr;
    return local_addr;
}

void DataManager::cacheCoolDown(size_t) {}

void DataManager::cacheEvict(size_t) {}

bool DataManager::evict(MetaData* metadata) {
    return try_evict_hot_leaf(metadata);
}

void DataManager::markDirty(GlobalAddr gaddr) {
    if (!gaddr.global_tag) {
        return;
    }
    MetaData* metadata = nullptr;
    if (gaddr.local_tag) {
        auto metadata_addr = metadata_addr_from_local_addr(reinterpret_cast<void*>(gaddr.addr));
        metadata = reinterpret_cast<MetaData*>(metadata_addr->get_addr());
    } else {
        metadata = reinterpret_cast<MetaData*>(gaddr.addr);
    }
    if (metadata != nullptr) {
        metadata->dirty.store(true, std::memory_order_relaxed);
    }
}

void DataManager::cacheDirectEvict(size_t need_size) {
    int tid = get_thread_id();
    size_t evicted_size = 0;
    int retry_count = 0;
    constexpr int kRetryLimit = 20000;

    while (evicted_size < need_size) {
        MetaData* metadata = owner_lru.pop_head_candidate(tid);
        if (metadata == nullptr) {
            publish_quiescent_epoch(tid);
            reclaim_cooling_addrs(tid);
            if (!evicted_addrs[tid].empty() || !cooling_addrs[tid].empty()) {
                return;
            }
            return;
        }

        ++retry_count;
        if (retry_count > kRetryLimit) {
            return;
        }

        if (cache_children_num_load(metadata) > 0 ||
            metadata->get_state() != state_in_cache_hot || metadata->local_addr == nullptr) {
            owner_lru.release_owner(metadata);
            continue;
        }

        auto aligned = static_cast<size_t>((metadata->size + sizeof(MetaDataAddr) + 63) & ~63);
        if (try_evict_hot_leaf(metadata)) {
            evicted_size += aligned;
            continue;
        }

        if (metadata->owner_tid.load(std::memory_order_acquire) == tid &&
            cache_children_num_load(metadata) == 0 &&
            metadata->get_state() == state_in_cache_hot && metadata->local_addr != nullptr) {
            owner_lru.offer(metadata, tid, state_in_cache_hot);
        } else {
            owner_lru.release_owner(metadata);
        }
    }
}

void DataManager::releaseCurrentThread() {
    int tid = get_thread_id();
    last_cxl_access[tid * STRIDE].store(nullptr, std::memory_order_seq_cst);
    publish_quiescent_epoch(tid);
    reclaim_cooling_addrs(tid);
    epoch_states[tid * STRIDE].active.store(false, std::memory_order_release);
    owner_lru.release_thread(tid);

}

extern "C" {
void pth_bm_target_print_stat(void*) {
    std::cout << "malloc size: " << global_state.data->malloc_size
              << " bytes, count: " << global_state.data->malloc_count << std::endl;
#ifdef PENTATHLON_USE_CXL
    std::cout << "CXL read: " << global_state.rdma->read_count
              << ", CXL write: " << global_state.rdma->write_count << std::endl;
#else
    std::cout << "RDMA read: " << global_state.rdma->read_count
              << ", RDMA write: " << global_state.rdma->write_count << std::endl;
#endif
}

void pth_bm_target_reset_data_manager() {
    global_state.data->thread_count.store(0);
    global_state.data->reset_epoch_participants();
    // printf("evicted_addrs size for thread 0: %zu\n", global_state.data->evicted_addrs[0].size());
}

void pth_bm_target_release_thread() {
    global_state.data->releaseCurrentThread();
}
}
