add_executable(dm_compiler_rt_memory
    src/context.c
    src/main.c
    src/rdma.c
)
target_include_directories(dm_compiler_rt_memory
    PRIVATE
        src
        ../common
        ${IBVERBS_INCLUDE_DIRS}
)
target_link_libraries(dm_compiler_rt_memory RDMA::RDMAcm Threads::Threads ${IBVERBS_LIBRARIES})
