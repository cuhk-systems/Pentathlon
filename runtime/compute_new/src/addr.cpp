#include "addr.h"
#include "init.hpp"

void* getLocalAddr(void* ptr) {
    // Implementation of getting local address
    // This function should return the local address associated with the given global address
    // fprintf(stderr, "Get local address for pointer: %p\n", ptr);
    // assert((((uint64_t)ptr & 0xffffffff00000000ull) != 0x0000555500000000ull) && "heap memory should not show in getLocalAddr");
    if (isLocalAddr(ptr)) {
        return ptr;
    }
    GlobalAddr gaddr = GlobalAddr::fromPointer(ptr);  // Cast the pointer to GlobalAddr
    if (!gaddr.global_tag) {
        return ptr;
    }
    void* local_addr = global_state.data->getLocalAddr(gaddr);
    if (local_addr == nullptr) {
        throw std::runtime_error("Failed to get local address");
    }
    return local_addr;
}

void* updateAddrDep(void* father_ptr, void* child_ptr) {
    // Implementation of updateAddrDep
    // This function should handle the address dependency for the given address
    // return child_ptr;
    // return;
    GlobalAddr father = GlobalAddr::fromPointer(father_ptr);  // Cast the pointer to GlobalAddr
    GlobalAddr child = GlobalAddr::fromPointer(child_ptr);  // Cast the pointer to GlobalAddr
    // if (!father.global_tag || !father.local_tag) {
    //     throw std::runtime_error("Invalid father address in updateAddrDep");
    // }
    // return;
    if (!child.global_tag || !child.local_tag) {
        return child_ptr;
    }
    // MetaData* metadata = *reinterpret_cast<MetaData**>(gaddr.addr - 8);
    // auto new_gaddr = GlobalAddr{gaddr.offset, reinterpret_cast<uint64_t>(metadata), 0, 1};
    return global_state.data->updateAddrDep(father, child);
}

bool isLocalAddr(void* ptr) {
    // Implementation of checking if the address is local
    // This function should return true if the address is local, false otherwise
    GlobalAddr gaddr = GlobalAddr::fromPointer(ptr);  // Cast the pointer to GlobalAddr
    return !gaddr.global_tag;
}

void releaseLocalAddr(void* ptr) {
    // Implementation of releasing address
    // This function should handle the releasing of the memory associated with the given address
    // GlobalAddr gaddr = GlobalAddr::fromPointer(ptr);  // Cast the pointer to GlobalAddr
    // if (!gaddr.global_tag) {
    //     return;
    // }
    // global_state.data->releaseLocalAddr(gaddr);
}

void addAddrDep(void* addr_u, void* addr_v) {
    // Implementation of adding address dependency
    // debug print
    // fprintf(stderr, "Add address dependency: %p -> %p\n", addr_u, addr_v);
    global_state.data->addAddrDep(GlobalAddr::fromPointer(addr_u), GlobalAddr::fromPointer(addr_v));
}

void markDirty(void* addr) {
    // Implementation of marking address as dirty
    // fprintf(stderr, "Mark address as dirty: %p\n", addr);
    global_state.data->markDirty(GlobalAddr::fromPointer(addr));
}

void addAddrDepDebug(void* addr_u, void* addr_v) {
    // Implementation of adding address dependency for debug
    // debug print
    // fprintf(stderr, "Add address dependency (debug): %p <- %p\n", addr_u, addr_v);
    // global_state.data->addAddrDep(GlobalAddr::fromPointer(addr_u), GlobalAddr::fromPointer(addr_v));
}
