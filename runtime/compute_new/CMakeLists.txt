set(DM_COMPILER_RT_COMPUTE_SOURCES
    src/addr.cpp
    src/alloc.cpp
    src/data-manager.cpp
    src/init.cpp
    src/lru.cpp
    src/pth_memset.c
)

if (PENTATHLON_USE_CXL)
    list(APPEND DM_COMPILER_RT_COMPUTE_SOURCES src/cxl.cpp)
else ()
    list(APPEND DM_COMPILER_RT_COMPUTE_SOURCES src/rdma.c src/rdma.cpp)
endif ()

add_library(dm_compiler_rt_compute
    ${DM_COMPILER_RT_COMPUTE_SOURCES}
)
target_include_directories(dm_compiler_rt_compute
    PUBLIC
        include
    # PRIVATE
        src
)
if (NOT PENTATHLON_USE_CXL)
    target_include_directories(dm_compiler_rt_compute
        PUBLIC
            ${IBVERBS_INCLUDE_DIRS}
    )
endif ()

if (PENTATHLON_USE_CXL)
    target_compile_definitions(dm_compiler_rt_compute PUBLIC PENTATHLON_USE_CXL=1)
endif ()

target_link_libraries(dm_compiler_rt_compute
    Threads::Threads
    numa
)
if (TARGET TBB::tbb)
    target_link_libraries(dm_compiler_rt_compute TBB::tbb)
endif ()
if (NOT PENTATHLON_USE_CXL)
    target_link_libraries(dm_compiler_rt_compute
        ${IBVERBS_LIBRARIES}
        RDMA::RDMAcm
    )
endif ()

add_executable(dm_compiler_rt_compute_tests
    tests/main.cpp
)
target_include_directories(dm_compiler_rt_compute_tests
    PRIVATE
        tests
)
# Link with threading support for multithreaded test
target_link_libraries(dm_compiler_rt_compute_tests
    dm_compiler_rt_compute
    Threads::Threads
)

# C version tester
add_executable(dm_compiler_rt_compute_tests_c
    tests/main_c.c
)
target_include_directories(dm_compiler_rt_compute_tests_c
    PRIVATE
        tests
)
# Link with threading support for multithreaded test (C version)
target_link_libraries(dm_compiler_rt_compute_tests_c
    dm_compiler_rt_compute
    Threads::Threads
)

add_executable(test-batch-read
    tests/batch_read.cpp
)
target_include_directories(test-batch-read
    PRIVATE
        tests
)
target_link_libraries(test-batch-read
    dm_compiler_rt_compute
    Threads::Threads
)

set(COMPUTE_NEW_TEST_TARGETS
    dm_compiler_rt_compute_tests
    dm_compiler_rt_compute_tests_c
    test-batch-read
)

add_custom_target(compute_new-test-binaries
    DEPENDS ${COMPUTE_NEW_TEST_TARGETS}
)
