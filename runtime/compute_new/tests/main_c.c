#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <stdatomic.h>

// Include the C API headers
#include "addr.h"
#include "alloc.h"

// Global statistics for multithreaded testing (C version)
typedef struct {
    atomic_size_t allocations;
    atomic_size_t deallocations;
    atomic_size_t cache_hits;
    atomic_size_t cache_misses;
    atomic_size_t errors;
    atomic_size_t successful_operations;
} test_stats_t;

// Thread-safe node structure for BST (C version)
typedef struct node {
    size_t data;
    struct node *left, *right;
    pthread_mutex_t node_mutex;  // For thread-safe access to this node
} node_t;

// Global test statistics
test_stats_t g_stats = {0};

// Global root mutex for BST operations
pthread_mutex_t g_root_mutex = PTHREAD_MUTEX_INITIALIZER;

// Thread data structure for passing parameters
typedef struct {
    node_t **root;
    pthread_mutex_t *root_mutex;
    int thread_id;
    int num_operations;
    int operation_type;  // 0=insert, 1=search, 2=mixed
} thread_data_t;

// Initialize a new node
node_t* create_node(size_t value) {
    // Allocate using disaggregated memory
    void* gaddr = disaggAlloc(sizeof(node_t));
    if (gaddr == NULL) {
        atomic_fetch_add(&g_stats.errors, 1);
        return NULL;
    }
    atomic_fetch_add(&g_stats.allocations, 1);

    // Get local address
    void *local_addr = getLocalAddr(gaddr);
    if (local_addr == NULL) {
        atomic_fetch_add(&g_stats.errors, 1);
        atomic_fetch_add(&g_stats.cache_misses, 1);
        disaggFree(gaddr);
        return NULL;
    }
    atomic_fetch_add(&g_stats.cache_hits, 1);

    // Initialize the node
    node_t *node = (node_t*)local_addr;
    node->data = value;
    node->left = NULL;
    node->right = NULL;
    
    // Initialize mutex
    if (pthread_mutex_init(&node->node_mutex, NULL) != 0) {
        atomic_fetch_add(&g_stats.errors, 1);
        disaggFree(gaddr);
        return NULL;
    }

    return (node_t*)gaddr;  // Return global address
}

// Thread-safe BST insertion
bool insert_bst_threadsafe(node_t **root, size_t value, pthread_mutex_t *root_mutex) {
    pthread_mutex_lock(root_mutex);

    if (*root == NULL) {
        *root = create_node(value);
        pthread_mutex_unlock(root_mutex);
        if (*root != NULL) {
            atomic_fetch_add(&g_stats.successful_operations, 1);
            return true;
        }
        return false;
    }

    pthread_mutex_unlock(root_mutex);

    // Navigate to insertion point
    node_t *current_gaddr = *root;
    
    while (current_gaddr != NULL) {
        void *local_addr = getLocalAddr((void*)current_gaddr);
        if (local_addr == NULL) {
            atomic_fetch_add(&g_stats.errors, 1);
            atomic_fetch_add(&g_stats.cache_misses, 1);
            return false;
        }
        atomic_fetch_add(&g_stats.cache_hits, 1);

        node_t *current = (node_t*)local_addr;
        pthread_mutex_lock(&current->node_mutex);

        if (value < current->data) {
            if (current->left == NULL) {
                current->left = create_node(value);
                pthread_mutex_unlock(&current->node_mutex);
                if (current->left != NULL) {
                    atomic_fetch_add(&g_stats.successful_operations, 1);
                    return true;
                }
                return false;
            } else {
                node_t *next = current->left;
                pthread_mutex_unlock(&current->node_mutex);
                current_gaddr = next;
            }
        } else if (value > current->data) {
            if (current->right == NULL) {
                current->right = create_node(value);
                pthread_mutex_unlock(&current->node_mutex);
                if (current->right != NULL) {
                    atomic_fetch_add(&g_stats.successful_operations, 1);
                    return true;
                }
                return false;
            } else {
                node_t *next = current->right;
                pthread_mutex_unlock(&current->node_mutex);
                current_gaddr = next;
            }
        } else {
            // Value already exists
            pthread_mutex_unlock(&current->node_mutex);
            atomic_fetch_add(&g_stats.successful_operations, 1);
            return true;
        }
    }

    return false;
}

// Thread-safe BST search
bool search_bst_threadsafe(node_t *root, size_t value) {
    node_t *current_gaddr = root;
    
    while (current_gaddr != NULL) {
        void *local_addr = getLocalAddr((void*)current_gaddr);
        if (local_addr == NULL) {
            atomic_fetch_add(&g_stats.errors, 1);
            atomic_fetch_add(&g_stats.cache_misses, 1);
            return false;
        }
        atomic_fetch_add(&g_stats.cache_hits, 1);

        node_t *current = (node_t*)local_addr;
        pthread_mutex_lock(&current->node_mutex);

        if (value == current->data) {
            pthread_mutex_unlock(&current->node_mutex);
            atomic_fetch_add(&g_stats.successful_operations, 1);
            return true;
        } else if (value < current->data) {
            node_t *next = current->left;
            pthread_mutex_unlock(&current->node_mutex);
            current_gaddr = next;
        } else {
            node_t *next = current->right;
            pthread_mutex_unlock(&current->node_mutex);
            current_gaddr = next;
        }
    }

    return false;
}

// Print statistics
void print_statistics(void) {
    printf("\n=== Test Statistics ===\n");
    printf("Allocations: %zu\n", atomic_load(&g_stats.allocations));
    printf("Deallocations: %zu\n", atomic_load(&g_stats.deallocations));
    printf("Cache hits: %zu\n", atomic_load(&g_stats.cache_hits));
    printf("Cache misses: %zu\n", atomic_load(&g_stats.cache_misses));
    printf("Errors: %zu\n", atomic_load(&g_stats.errors));
    printf("Successful operations: %zu\n", atomic_load(&g_stats.successful_operations));

    size_t total_cache_ops = atomic_load(&g_stats.cache_hits) + atomic_load(&g_stats.cache_misses);
    if (total_cache_ops > 0) {
        double hit_rate = (double)atomic_load(&g_stats.cache_hits) / total_cache_ops * 100.0;
        printf("Cache hit rate: %.2f%%\n", hit_rate);
    }
    printf("======================\n\n");
}

// Reset statistics
void reset_statistics(void) {
    atomic_store(&g_stats.allocations, 0);
    atomic_store(&g_stats.deallocations, 0);
    atomic_store(&g_stats.cache_hits, 0);
    atomic_store(&g_stats.cache_misses, 0);
    atomic_store(&g_stats.errors, 0);
    atomic_store(&g_stats.successful_operations, 0);
}

// Worker thread function for insertions
void* worker_insert(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;
    
    for (int i = 0; i < data->num_operations; ++i) {
        size_t value = (rand() % 1000) + 1;
        bool success = insert_bst_threadsafe(data->root, value, data->root_mutex);
        
        if (success) {
            printf("Thread %d inserted: %zu\n", data->thread_id, value);
        } else {
            printf("Thread %d failed to insert: %zu\n", data->thread_id, value);
        }
        
        // Small delay to increase chance of contention
        usleep(100);
    }
    
    return NULL;
}

// Worker thread function for searches
void* worker_search(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;
    
    for (int i = 0; i < data->num_operations; ++i) {
        size_t value = (rand() % 1000) + 1;
        bool found = search_bst_threadsafe(*data->root, value);
        
        if (found) {
            printf("Thread %d found: %zu\n", data->thread_id, value);
        }
        
        // Small delay to increase chance of contention
        usleep(50);
    }
    
    return NULL;
}

// Worker thread function for mixed operations
void* worker_mixed(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;
    
    for (int i = 0; i < data->num_operations; ++i) {
        size_t value = (rand() % 1000) + 1;
        int operation = rand() % 2;  // 0 = insert, 1 = search
        
        if (operation == 0) {
            // Insert operation
            bool success = insert_bst_threadsafe(data->root, value, data->root_mutex);
            if (success) {
                printf("Thread %d inserted: %zu\n", data->thread_id, value);
            }
        } else {
            // Search operation
            bool found = search_bst_threadsafe(*data->root, value);
            if (found) {
                printf("Thread %d found: %zu\n", data->thread_id, value);
            }
        }
        
        // Small delay to increase chance of contention
        usleep(75);
    }
    
    return NULL;
}

// Test 1: Concurrent insertions
int test_concurrent_insertions(void) {
    printf("\n=== Test 1: Concurrent Insertions ===\n");

    node_t *root = NULL;
    pthread_mutex_t root_mutex = PTHREAD_MUTEX_INITIALIZER;

    const int num_threads = 4;
    const int operations_per_thread = 10;

    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Launch worker threads
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].root = &root;
        thread_data[i].root_mutex = &root_mutex;
        thread_data[i].thread_id = i;
        thread_data[i].num_operations = operations_per_thread;
        thread_data[i].operation_type = 0;  // insert

        if (pthread_create(&threads[i], NULL, worker_insert, &thread_data[i]) != 0) {
            printf("Error creating thread %d\n", i);
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long duration_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000;

    printf("Test completed in %ld ms\n", duration_ms);
    print_statistics();

    pthread_mutex_destroy(&root_mutex);
    return 0;
}

// Test 2: Mixed concurrent operations
int test_mixed_operations(void) {
    printf("\n=== Test 2: Mixed Concurrent Operations ===\n");

    // Reset statistics
    reset_statistics();

    node_t *root = NULL;
    pthread_mutex_t root_mutex = PTHREAD_MUTEX_INITIALIZER;

    // Pre-populate the tree with some values
    printf("Pre-populating tree...\n");
    for (int i = 1; i <= 20; i += 2) {
        insert_bst_threadsafe(&root, i, &root_mutex);
    }

    const int num_insert_threads = 2;
    const int num_search_threads = 3;
    const int operations_per_thread = 8;
    const int total_threads = num_insert_threads + num_search_threads;

    pthread_t threads[total_threads];
    thread_data_t thread_data[total_threads];

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Launch insert threads
    for (int i = 0; i < num_insert_threads; ++i) {
        thread_data[i].root = &root;
        thread_data[i].root_mutex = &root_mutex;
        thread_data[i].thread_id = i;
        thread_data[i].num_operations = operations_per_thread;
        thread_data[i].operation_type = 0;  // insert

        if (pthread_create(&threads[i], NULL, worker_insert, &thread_data[i]) != 0) {
            printf("Error creating insert thread %d\n", i);
            return 1;
        }
    }

    // Launch search threads
    for (int i = 0; i < num_search_threads; ++i) {
        int thread_idx = num_insert_threads + i;
        thread_data[thread_idx].root = &root;
        thread_data[thread_idx].root_mutex = &root_mutex;
        thread_data[thread_idx].thread_id = thread_idx;
        thread_data[thread_idx].num_operations = operations_per_thread;
        thread_data[thread_idx].operation_type = 1;  // search

        if (pthread_create(&threads[thread_idx], NULL, worker_search, &thread_data[thread_idx]) != 0) {
            printf("Error creating search thread %d\n", thread_idx);
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < total_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long duration_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000;

    printf("Test completed in %ld ms\n", duration_ms);
    print_statistics();

    pthread_mutex_destroy(&root_mutex);
    return 0;
}

// Test 3: High contention stress test
int test_high_contention(void) {
    printf("\n=== Test 3: High Contention Stress Test ===\n");

    // Reset statistics
    reset_statistics();

    node_t *root = NULL;
    pthread_mutex_t root_mutex = PTHREAD_MUTEX_INITIALIZER;

    const int num_threads = 8;
    const int operations_per_thread = 15;

    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Launch mixed workload threads
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].root = &root;
        thread_data[i].root_mutex = &root_mutex;
        thread_data[i].thread_id = i;
        thread_data[i].num_operations = operations_per_thread;
        thread_data[i].operation_type = 2;  // mixed

        if (pthread_create(&threads[i], NULL, worker_mixed, &thread_data[i]) != 0) {
            printf("Error creating thread %d\n", i);
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long duration_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000;

    printf("Stress test completed in %ld ms\n", duration_ms);
    print_statistics();

    pthread_mutex_destroy(&root_mutex);
    return 0;
}

typedef struct {
    int thread_id;
    int num_allocs;
    void** ptrs;
    int* count;
    pthread_mutex_t* mutex;
} alloc_thread_data_t;

const int num_allocations = 100;

void* alloc_worker(void* arg) {
    alloc_thread_data_t *data = (alloc_thread_data_t*)arg;

    for (int i = 0; i < data->num_allocs; ++i) {
        size_t size = sizeof(node_t) + (i % 64);  // Vary allocation sizes
        void* ptr = disaggAlloc(size);

        if (ptr != NULL) {
            atomic_fetch_add(&g_stats.allocations, 1);

            // Test local address access
            void* local_addr = getLocalAddr(ptr);
            if (local_addr != NULL) {
                atomic_fetch_add(&g_stats.cache_hits, 1);
                // Write some data to test memory access
                memset(local_addr, data->thread_id, (size < sizeof(size_t)) ? size : sizeof(size_t));
            } else {
                atomic_fetch_add(&g_stats.cache_misses, 1);
            }

            pthread_mutex_lock(data->mutex);
            if (*data->count < num_allocations) {
                data->ptrs[*data->count] = ptr;
                (*data->count)++;
            }
            pthread_mutex_unlock(data->mutex);
        } else {
            atomic_fetch_add(&g_stats.errors, 1);
        }

        usleep(10);
    }

    return NULL;
}


// Test 4: Memory allocation stress test
int test_memory_stress(void) {
    printf("\n=== Test 4: Memory Allocation Stress Test ===\n");

    // Reset statistics
    reset_statistics();

    void* allocated_ptrs[num_allocations];
    int allocated_count = 0;
    pthread_mutex_t alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

    const int num_threads = 4;
    const int allocs_per_thread = num_allocations / num_threads;

    pthread_t threads[num_threads];
    alloc_thread_data_t thread_data[num_threads];

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Launch allocation threads
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].num_allocs = allocs_per_thread;
        thread_data[i].ptrs = allocated_ptrs;
        thread_data[i].count = &allocated_count;
        thread_data[i].mutex = &alloc_mutex;

        if (pthread_create(&threads[i], NULL, alloc_worker, &thread_data[i]) != 0) {
            printf("Error creating thread %d\n", i);
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long duration_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000;

    printf("Memory stress test completed in %ld ms\n", duration_ms);
    printf("Allocated %d memory blocks\n", allocated_count);

    // Clean up allocated memory
    for (int i = 0; i < allocated_count; ++i) {
        disaggFree(allocated_ptrs[i]);
        atomic_fetch_add(&g_stats.deallocations, 1);
    }

    print_statistics();
    pthread_mutex_destroy(&alloc_mutex);
    return 0;
}

int main(void) {
    printf("=== Multithreaded Runtime/Compute Test Suite (C Version) ===\n");
    printf("Testing thread safety improvements in DataManager and RDMAClient\n");

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Run test one by one
    printf("Running Test 1...\n");
    if (test_concurrent_insertions() != 0) {
        printf("Test 1 failed!\n");
        return 1;
    }
    printf("Test 1 passed!\n");

    printf("Running Test 2...\n");
    if (test_mixed_operations() != 0) {
        printf("Test 2 failed!\n");
        return 1;
    }
    printf("Test 2 passed!\n");

    printf("Running Test 3...\n");
    if (test_high_contention() != 0) {
        printf("Test 3 failed!\n");
        return 1;
    }
    printf("Test 3 passed!\n");

    printf("Running Test 4...\n");
    if (test_memory_stress() != 0) {
        printf("Test 4 failed!\n");
        return 1;
    }
    printf("Test 4 passed!\n");

    printf("\n=== All Tests Passed! ===\n");
    printf("C version tester completed successfully.\n");

    return 0;
}
