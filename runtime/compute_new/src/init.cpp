#include <cstdlib>
#include <stdexcept>
#include <stdio.h>

#include "init.h"
#include "init.hpp"
#include "rdma.hpp"

GlobalState::GlobalState() {
#ifdef PENTATHLON_USE_CXL
    std::printf("CXL Init started.\n");
    sockaddr_storage addr = {};
    rdma = new RDMAClient(addr);
#else
    // Initialize RDMA
    char* addr_str = getenv("MEMORY_ADDR");
    if (!addr_str) {
        throw std::runtime_error(
            "please set MEMORY_ADDR environment variable to memory server address");
    }
    sockaddr_storage addr = {};
    if (c_parse_addr(addr_str, &addr, 12345) < 0) {
        throw std::runtime_error("failed to parse network address");
    }
    std::printf("RDMA Init started.\n");
    rdma = new RDMAClient(addr);

    std::printf("RDMA Init finished.\n");
#endif

    // Initialize data manager
    data = new DataManager(rdma);

    std::printf("DataManager Init finished.\n");


    // Initialize other members...
}
// GlobalState destructor implementation
GlobalState::~GlobalState() {
    delete rdma;
    delete data;
    // Clean up other members
}

// // Static method to get the single instance (Meyers' Singleton)
// GlobalState& GlobalState::get_instance() {
//     // The 'static' variable 'instance' is constructed only once,
//     // upon the first call to GlobalState::get_instance().
//     // This initialization is guaranteed to be thread-safe from C++11 onwards.
//     static GlobalState instance;
//     return instance;
// }


GlobalState global_state = GlobalState();

extern "C" {
void __ensureUsed() {}
}
