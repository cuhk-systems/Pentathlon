#ifndef COMPUTE_GLOBAL_ADDR_H
#define COMPUTE_GLOBAL_ADDR_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

union GlobalAddr {
    struct {
        uint32_t offset : 14;
        uint64_t addr : 48;
        uint8_t local_tag : 1;
        uint8_t global_tag : 1;
    };
    uint64_t val;

#ifdef __cplusplus
    static GlobalAddr null() { return {.val = 0}; }
    bool operator==(const GlobalAddr& other) const { return val == other.val; }
    bool operator!=(const GlobalAddr& other) const { return val != other.val; }
    bool operator<(const GlobalAddr& other) const { return val < other.val; }
    // transform to a pointer type
    operator void*() const {
        return reinterpret_cast<void*>(val);
    }
    // transform from a pointer type
    static GlobalAddr fromPointer(void* ptr) {
        GlobalAddr gaddr;
        gaddr.val = reinterpret_cast<uint64_t>(ptr);
        return gaddr;
    }
#endif
};

void* getLocalAddr(void* gaddr);
void* updateAddrDep(void* father_ptr, void* child_ptr);
bool isLocalAddr(void* gaddr);
void releaseLocalAddr(void* gaddr);

void addAddrDep(void* addr_u, void* addr_v);
void markDirty(void* addr);
void addAddrDepDebug(void* addr_u, void* addr_v);

#ifdef __cplusplus
}

#include <unordered_map>

// Provide hash and equality for global_addr_t if not already defined
namespace std {
template <>
struct hash<GlobalAddr> {
    std::size_t operator()(const GlobalAddr& g) const noexcept {
        // Replace this with an appropriate hash for global_addr_t
        // Assuming global_addr_t has a member 'addr' of type uint64_t
        return std::hash<uint64_t>()(g.val);
    }
};
template <>
struct equal_to<GlobalAddr> {
    bool operator()(const GlobalAddr& lhs, const GlobalAddr& rhs) const noexcept {
        // Replace this with an appropriate equality check for global_addr_t
        return lhs.val == rhs.val;
    }
};
}  // namespace std
#endif

#endif  // _COMPUTE_GLOBAL_ADDR_H_
