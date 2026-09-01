#ifndef COMPUTE_METADATA_HPP
#define COMPUTE_METADATA_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

struct LruNode;

struct alignas(8) MetaDataAddr {
    std::atomic<uint8_t> state;
    uint8_t addr_padding[7];
    std::atomic<int> cache_children_num;

    MetaDataAddr() : state(0), cache_children_num(0) {
        std::memset(addr_padding, 0, 7);
    }

    void* get_addr() {
        uint64_t addr = 0;
        std::memcpy(&addr, addr_padding, 6);
        return reinterpret_cast<void*>(addr);
    }

    void set_addr(uint64_t addr) {
        if (addr >> 48) {
            throw std::runtime_error("address exceeds 48 bits");
        }
        std::memcpy(addr_padding, &addr, 6);
    }

    void reset() {
        state.store(0);
        cache_children_num.store(0, std::memory_order_relaxed);
        std::memset(addr_padding, 0, 7);
    }
};

static_assert(sizeof(MetaDataAddr) == 16, "MetaDataAddr is the per-object cache header");

inline MetaDataAddr* metadata_addr_from_local_addr(void* local_addr) {
    return reinterpret_cast<MetaDataAddr*>((char*)local_addr - sizeof(MetaDataAddr));
}

struct alignas(64) MetaData {
    void* local_addr;
    void* remote_addr;
    std::atomic<MetaData*> father;
    std::atomic<int> father_offset;
    std::atomic<bool> dirty;
    std::atomic<bool> loading_flag;
    std::atomic<int> owner_tid;
    std::atomic<uint64_t> cooling_epoch;
    std::atomic<uint64_t> access_count;
    LruNode* lru_node;
    int size;

    MetaData() : local_addr(nullptr),
                 remote_addr(nullptr),
                 father(nullptr),
                 father_offset(0),
                 dirty(false),
                 loading_flag(false),
                 owner_tid(-1),
                 access_count(0),
                 cooling_epoch(0),
                 lru_node(nullptr),
                 size(0) {}

    inline int get_state() {
        if (local_addr == nullptr) {
            return 0;
        }
        auto* metadata_addr = metadata_addr_from_local_addr(local_addr);
        return metadata_addr->state.load();
    }

    inline int cache_children_num_in_state(int hot_state) {
        if (local_addr != nullptr) {
            auto* metadata_addr = metadata_addr_from_local_addr(local_addr);
            if (reinterpret_cast<MetaData*>(metadata_addr->get_addr()) == this &&
                metadata_addr->state.load(std::memory_order_relaxed) == hot_state) {
                return metadata_addr->cache_children_num.load(std::memory_order_relaxed);
            }
        }
        return 0;
    }
};

#endif
