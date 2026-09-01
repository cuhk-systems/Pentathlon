#include <cstdio>

#include "addr.h"
#include "alloc.h"
#include "init.hpp"

void* disaggAlloc(size_t size) {
    // Implementation of disaggregated allocation
    // This function should handle the allocation of memory of the given size
    // It should return a global address that represents the allocated memory
    GlobalAddr gaddr;
    gaddr.val = 0;  // Placeholder for actual allocation logic
    gaddr = global_state.data->disaggAlloc(size);  // Call the disaggregated allocation function
    if (gaddr.val == 0) {
        fprintf(stderr, "Failed to allocate memory of size: %zu\n", size);
        return GlobalAddr::null();  // Return null if allocation fails
    }
    // printf("disaggAlloc size = %lu, gaddr = %lx\n", size, gaddr.val);
    return gaddr;  // Return the allocated global address
}

void disaggFree(void* ptr) {
    // Implementation of disaggregated free
    // This function should handle the freeing of the memory associated with the given global
    // address It should also update the cache and any necessary data structures Check if the
    // address is valid
    // GlobalAddr gaddr = GlobalAddr::fromPointer(ptr);  // Cast the pointer to GlobalAddr
    // if (gaddr.val == 0) {
    //     fprintf(stderr, "Invalid global address: %lu\n", gaddr.val);
    //     return;  // Invalid address, do nothing
    // }
    // global_state.data->disaggFree(gaddr);  // Call the disaggregated free function
    // if (gaddr.val == 0) {
    //     fprintf(stderr, "Failed to free memory for global address: %lu\n", gaddr.val);
    //     return;  // Return on failure
    // }
}
