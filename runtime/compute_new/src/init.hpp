#ifndef COMPUTE_INIT_HPP
#define COMPUTE_INIT_HPP

#include "data-manager.hpp"
#include "rdma.hpp"

struct GlobalState {
    RDMAClient* rdma;
    DataManager* data;
    // other members...

    GlobalState();
    GlobalState(const GlobalState&) = delete;
    GlobalState& operator=(const GlobalState&) = delete;
    // Static method to get the single instance of GlobalState
    static GlobalState& get_instance();

    // Destructor (important for cleaning up dynamically allocated members)
    ~GlobalState();
};

extern GlobalState global_state;

#endif
