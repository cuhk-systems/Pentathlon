#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <memory>
#include <cassert>
#include <functional>
#include <cstring>
#include <string>
#include "addr.h"
#include "alloc.h"
#include "init.hpp"
#include "rdma.hpp"

// Global statistics for multithreaded testing
struct TestStats {
    std::atomic<size_t> allocations{0};
    std::atomic<size_t> deallocations{0};
    std::atomic<size_t> cache_hits{0};
    std::atomic<size_t> cache_misses{0};
    std::atomic<size_t> errors{0};
    std::atomic<size_t> successful_operations{0};
    void reset() {
        allocations = 0;
        deallocations = 0;
        cache_hits = 0;
        cache_misses = 0;
        errors = 0;
        successful_operations = 0;
    }
};

// Thread-safe node structure for BST
struct node {
    size_t data;
    node *left, *right;
    std::mutex node_mutex;  // For thread-safe access to this node

    node(size_t val) : data(val), left(nullptr), right(nullptr) {}
};

// Global test statistics
TestStats g_stats;

// Thread-safe random number generator
thread_local std::mt19937 tl_rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Recursive insertion helper (assumes root is not null)
bool insertBST_recursive(node *current_gaddr, size_t value) {
    void *local_addr = getLocalAddr(reinterpret_cast<void *>(current_gaddr));
    if (local_addr == nullptr) {
        g_stats.errors++;
        g_stats.cache_misses++;
        return false;
    }
    g_stats.cache_hits++;

    node *current = reinterpret_cast<node *>(local_addr);
    std::lock_guard<std::mutex> node_lock(current->node_mutex);

    if (value < current->data) {
        if (current->left == nullptr) {
            // Create new left child
            void* gaddr = disaggAlloc(sizeof(node));
            if (gaddr == nullptr) {
                g_stats.errors++;
                return false;
            }
            g_stats.allocations++;

            current->left = reinterpret_cast<node *>(gaddr);
            void *left_local = getLocalAddr(reinterpret_cast<void *>(current->left));
            if (left_local == nullptr) {
                g_stats.errors++;
                g_stats.cache_misses++;
                return false;
            }
            g_stats.cache_hits++;

            new(left_local) node(value);
            g_stats.successful_operations++;
            return true;
        } else {
            return insertBST_recursive(current->left, value);
        }
    } else if (value > current->data) {
        if (current->right == nullptr) {
            // Create new right child
            void* gaddr = disaggAlloc(sizeof(node));
            if (gaddr == nullptr) {
                g_stats.errors++;
                return false;
            }
            g_stats.allocations++;

            current->right = reinterpret_cast<node *>(gaddr);
            void *right_local = getLocalAddr(reinterpret_cast<void *>(current->right));
            if (right_local == nullptr) {
                g_stats.errors++;
                g_stats.cache_misses++;
                return false;
            }
            g_stats.cache_hits++;

            new(right_local) node(value);
            g_stats.successful_operations++;
            return true;
        } else {
            return insertBST_recursive(current->right, value);
        }
    }

    // Value already exists, consider it successful
    g_stats.successful_operations++;
    return true;
}

// Thread-safe BST insertion with proper error handling
bool insertBST_threadsafe(node *&root, size_t value, std::mutex& root_mutex) {
    std::lock_guard<std::mutex> root_lock(root_mutex);

    if (root == nullptr) {
        // Allocate new node
        void* gaddr = disaggAlloc(sizeof(node));
        if (gaddr == nullptr) {
            g_stats.errors++;
            return false;
        }
        g_stats.allocations++;

        root = reinterpret_cast<node *>(gaddr);
        void *local_addr = getLocalAddr(reinterpret_cast<void *>(root));
        if (local_addr == nullptr) {
            g_stats.errors++;
            g_stats.cache_misses++;
            return false;
        }
        g_stats.cache_hits++;

        // Initialize the new node using placement new
        node *current = new(local_addr) node(value);
        g_stats.successful_operations++;
        return true;
    }

    // Navigate to insertion point (recursive approach with locks)
    return insertBST_recursive(root, value);
}

// Thread-safe search function
bool searchBST_threadsafe(node *root, size_t value) {
    if (root == nullptr) {
        return false;
    }

    void *local_addr = getLocalAddr(reinterpret_cast<void *>(root));
    if (local_addr == nullptr) {
        g_stats.errors++;
        g_stats.cache_misses++;
        return false;
    }
    g_stats.cache_hits++;

    node *current = reinterpret_cast<node *>(local_addr);
    std::lock_guard<std::mutex> node_lock(current->node_mutex);

    if (value == current->data) {
        g_stats.successful_operations++;
        return true;
    } else if (value < current->data) {
        return searchBST_threadsafe(current->left, value);
    } else {
        return searchBST_threadsafe(current->right, value);
    }
}

// Thread-safe in-order traversal with callback
void traverseInOrder_threadsafe(node *root, std::function<void(size_t)> callback) {
    if (root != nullptr) {
        void *local_addr = getLocalAddr(reinterpret_cast<void *>(root));
        if (local_addr == nullptr) {
            g_stats.errors++;
            g_stats.cache_misses++;
            return;
        }
        g_stats.cache_hits++;

        node *current = reinterpret_cast<node *>(local_addr);

        // Store children pointers and data while holding lock
        node *left_child, *right_child;
        size_t data;

        {
            std::lock_guard<std::mutex> node_lock(current->node_mutex);
            left_child = current->left;
            right_child = current->right;
            data = current->data;
        }

        // Now traverse without holding the lock
        traverseInOrder_threadsafe(left_child, callback);
        callback(data);
        traverseInOrder_threadsafe(right_child, callback);
    }
}

// Simple print function for traversal
void printInOrder_threadsafe(node *root) {
    std::mutex print_mutex;
    traverseInOrder_threadsafe(root, [&print_mutex](size_t value) {
        std::lock_guard<std::mutex> print_lock(print_mutex);
        std::cout << value << " ";
    });
}

void maybe_log_operation(bool verbose, const std::string& message) {
    if (verbose) {
        std::cout << message << std::endl;
    }
}

void maybe_sleep_for(std::chrono::microseconds duration) {
    if (duration.count() > 0) {
        std::this_thread::sleep_for(duration);
    }
}

// Worker thread function for concurrent insertions
void worker_insert(node *&root,
                   std::mutex& root_mutex,
                   int thread_id,
                   int num_operations,
                   bool verbose = true,
                   std::chrono::microseconds sleep_duration = std::chrono::microseconds(100)) {
    std::uniform_int_distribution<int> dist(1, 1000);

    for (int i = 0; i < num_operations; ++i) {
        size_t value = dist(tl_rng);
        bool success = insertBST_threadsafe(root, value, root_mutex);

        if (success) {
            maybe_log_operation(verbose, "Thread " + std::to_string(thread_id) + " inserted: " + std::to_string(value));
        } else {
            maybe_log_operation(verbose, "Thread " + std::to_string(thread_id) + " failed to insert: " + std::to_string(value));
        }

        // Small delay to increase chance of contention
        maybe_sleep_for(sleep_duration);
    }
}

// Worker thread function for concurrent searches
void worker_search(node *root,
                   int thread_id,
                   int num_operations,
                   bool verbose = true,
                   std::chrono::microseconds sleep_duration = std::chrono::microseconds(50)) {
    std::uniform_int_distribution<int> dist(1, 1000);

    for (int i = 0; i < num_operations; ++i) {
        size_t value = dist(tl_rng);
        bool found = searchBST_threadsafe(root, value);

        if (found) {
            maybe_log_operation(verbose, "Thread " + std::to_string(thread_id) + " found: " + std::to_string(value));
        }

        // Small delay to increase chance of contention
        maybe_sleep_for(sleep_duration);
    }
}

// Mixed workload worker (insert + search)
void worker_mixed(node *&root,
                  std::mutex& root_mutex,
                  int thread_id,
                  int num_operations,
                  bool verbose = true,
                  std::chrono::microseconds sleep_duration = std::chrono::microseconds(75)) {
    std::uniform_int_distribution<int> dist(1, 1000);
    std::uniform_int_distribution<int> op_dist(0, 1);  // 0 = insert, 1 = search

    for (int i = 0; i < num_operations; ++i) {
        size_t value = dist(tl_rng);
        int operation = op_dist(tl_rng);

        if (operation == 0) {
            // Insert operation
            bool success = insertBST_threadsafe(root, value, root_mutex);
            if (success) {
                maybe_log_operation(verbose, "Thread " + std::to_string(thread_id) + " inserted: " + std::to_string(value));
            }
        } else {
            // Search operation
            bool found = searchBST_threadsafe(root, value);
            if (found) {
                maybe_log_operation(verbose, "Thread " + std::to_string(thread_id) + " found: " + std::to_string(value));
            }
        }

        // Small delay to increase chance of contention
        maybe_sleep_for(sleep_duration);
    }
}

// Print test statistics
void print_statistics() {
    std::cout << "\n=== Test Statistics ===" << std::endl;
    std::cout << "Allocations: " << g_stats.allocations.load() << std::endl;
    std::cout << "Deallocations: " << g_stats.deallocations.load() << std::endl;
    std::cout << "Cache hits: " << g_stats.cache_hits.load() << std::endl;
    std::cout << "Cache misses: " << g_stats.cache_misses.load() << std::endl;
    std::cout << "Errors: " << g_stats.errors.load() << std::endl;
    std::cout << "Successful operations: " << g_stats.successful_operations.load() << std::endl;

    size_t total_cache_ops = g_stats.cache_hits.load() + g_stats.cache_misses.load();
    if (total_cache_ops > 0) {
        double hit_rate = (double)g_stats.cache_hits.load() / total_cache_ops * 100.0;
        std::cout << "Cache hit rate: " << hit_rate << "%" << std::endl;
    }
    std::cout << "======================\n" << std::endl;
}

// Test 1: Concurrent insertions
void test_concurrent_insertions() {
    std::cout << "\n=== Test 1: Concurrent Insertions ===" << std::endl;

    node *root = nullptr;
    std::mutex root_mutex;

    const int num_threads = 4;
    const int operations_per_thread = 10;

    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            worker_insert,
            std::ref(root),
            std::ref(root_mutex),
            i,
            operations_per_thread,
            true,
            std::chrono::microseconds(100)
        );
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Test completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Final tree (in-order): ";
    printInOrder_threadsafe(root);
    std::cout << std::endl;

    print_statistics();
}

// Test 5: Large-scale mixed workload with more operations per thread
void test_large_scale_operations() {
    std::cout << "\n=== Test 5: Large-Scale Mixed Workload ===" << std::endl;

    g_stats.reset();

    node *root = nullptr;
    std::mutex root_mutex;

    const int num_threads = 4;
    const int operations_per_thread = 10000;

    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            worker_mixed,
            std::ref(root),
            std::ref(root_mutex),
            i,
            operations_per_thread,
            false,
            std::chrono::microseconds(0)
        );
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Large-scale workload completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Threads: " << num_threads << ", operations per thread: " << operations_per_thread << std::endl;

    print_statistics();
}

// Test 2: Mixed concurrent operations
void test_mixed_operations() {
    std::cout << "\n=== Test 2: Mixed Concurrent Operations ===" << std::endl;

    // Reset statistics
    g_stats.reset();

    node *root = nullptr;
    std::mutex root_mutex;

    // Pre-populate the tree with some values
    std::cout << "Pre-populating tree..." << std::endl;
    for (int i = 1; i <= 20; i += 2) {
        insertBST_threadsafe(root, i, root_mutex);
    }

    const int num_insert_threads = 2;
    const int num_search_threads = 3;
    const int operations_per_thread = 8;

    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch insert threads
    for (int i = 0; i < num_insert_threads; ++i) {
        threads.emplace_back(
            worker_insert,
            std::ref(root),
            std::ref(root_mutex),
            i,
            operations_per_thread,
            true,
            std::chrono::microseconds(100)
        );
    }

    // Launch search threads
    for (int i = 0; i < num_search_threads; ++i) {
        threads.emplace_back(
            worker_search,
            root,
            i + num_insert_threads,
            operations_per_thread,
            true,
            std::chrono::microseconds(50)
        );
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Test completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Final tree (in-order): ";
    printInOrder_threadsafe(root);
    std::cout << std::endl;

    print_statistics();
}

// Test 3: High contention stress test
void test_high_contention() {
    std::cout << "\n=== Test 3: High Contention Stress Test ===" << std::endl;

    // Reset statistics
    g_stats.reset();

    node *root = nullptr;
    std::mutex root_mutex;

    const int num_threads = 8;
    const int operations_per_thread = 15;

    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch mixed workload threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            worker_mixed,
            std::ref(root),
            std::ref(root_mutex),
            i,
            operations_per_thread,
            true,
            std::chrono::microseconds(75)
        );
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Stress test completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Final tree (in-order): ";
    printInOrder_threadsafe(root);
    std::cout << std::endl;

    print_statistics();
}

// Test 4: Memory allocation stress test
void test_memory_stress() {
    std::cout << "\n=== Test 4: Memory Allocation Stress Test ===" << std::endl;

    // Reset statistics
    g_stats.reset();

    const int num_allocations = 100;
    std::vector<void*> allocated_ptrs;
    std::mutex alloc_mutex;

    auto alloc_worker = [&](int thread_id, int num_allocs) {
        for (int i = 0; i < num_allocs; ++i) {
            size_t size = sizeof(node) + (i % 64);  // Vary allocation sizes
            void* ptr = disaggAlloc(size);

            if (ptr != nullptr) {
                g_stats.allocations++;

                // Test local address access
                void* local_addr = getLocalAddr(ptr);
                if (local_addr != nullptr) {
                    g_stats.cache_hits++;
                    // Write some data to test memory access
                    memset(local_addr, thread_id, std::min(size, sizeof(size_t)));
                } else {
                    g_stats.cache_misses++;
                }

                std::lock_guard<std::mutex> lock(alloc_mutex);
                allocated_ptrs.push_back(ptr);
            } else {
                g_stats.errors++;
            }

            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    const int num_threads = 4;
    const int allocs_per_thread = num_allocations / num_threads;

    std::vector<std::thread> threads;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch allocation threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(alloc_worker, i, allocs_per_thread);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Memory stress test completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Allocated " << allocated_ptrs.size() << " memory blocks" << std::endl;

    // Clean up allocated memory
    for (void* ptr : allocated_ptrs) {
        disaggFree(ptr);
        g_stats.deallocations++;
    }

    print_statistics();
}

int main() {
    std::cout << "=== Multithreaded Runtime/Compute Test Suite ===" << std::endl;
    std::cout << "Testing thread safety improvements in DataManager and RDMAClient" << std::endl;


    // run test one by one
    try {
        // Run all test scenarios
        test_concurrent_insertions();

        std::cout << "Test 1 passed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test 1 failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test 1 failed with unknown exception" << std::endl;
        return 1;
    }

    try {
        test_mixed_operations();
        std::cout << "Test 2 passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test 2 failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test 2 failed with unknown exception" << std::endl;
        return 1;
    }

    try {
        test_high_contention();
        std::cout << "Test 3 passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test 3 failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test 3 failed with unknown exception" << std::endl;
        return 1;
    }

    try {
        test_memory_stress();
        std::cout << "Test 4 passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test 4 failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test 4 failed with unknown exception" << std::endl;
        return 1;
    }

    try {
        test_large_scale_operations();
        std::cout << "Test 5 passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test 5 failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test 5 failed with unknown exception" << std::endl;
        return 1;
    }

    return 0;
}
