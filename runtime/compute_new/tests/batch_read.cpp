#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "../src/init.hpp"
#include "../src/rdma.hpp"
#include <chrono>

struct Node {
    uint64_t buf[128];
};

int main() {
    auto rdma = global_state.rdma;

    for (int i = 0; i < 100000; i++) {
        Node node;
        uint64_t addr = rdma->mem().addr + i * sizeof(Node);
        for (int j = 0; j < 128; j++) {
            node.buf[j] = ((uint64_t)rand() << 32) + rand();
        }
        rdma->write(&node, (void*)addr, sizeof(Node));
    }

    sleep(1);

    uint64_t time_batch = 0;
    for (int i = 99999; i >= 0; i -= 2) {
        Node node1, node2;
        auto pos1 = rand() % 100000;
        auto pos2 = rand() % 100000;
        uint64_t addr1 = rdma->mem().addr + pos1 * sizeof(Node);
        uint64_t addr2 = rdma->mem().addr + pos2 * sizeof(Node);
        // auto t1 = std::chrono::high_resolution_clock::now();
        rdma->read_batch_add((void*)addr1, &node1, (uint32_t)sizeof(Node));
        rdma->read_batch_add((void*)addr2, &node2, (uint32_t)sizeof(Node));
        rdma->batch_commit();
        // auto t2 = std::chrono::high_resolution_clock::now();
        // time_batch += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    for (int i = 99999; i >= 0; i -= 2) {
        Node node1, node2;
        auto pos1 = rand() % 100000;
        auto pos2 = rand() % 100000;
        uint64_t addr1 = rdma->mem().addr + pos1 * sizeof(Node);
        uint64_t addr2 = rdma->mem().addr + pos2 * sizeof(Node);
        auto t1 = std::chrono::high_resolution_clock::now();
        rdma->read_batch_add((void*)addr1, &node1, (uint32_t)sizeof(Node));
        rdma->read_batch_add((void*)addr2, &node2, (uint32_t)sizeof(Node));
        rdma->batch_commit();
        auto t2 = std::chrono::high_resolution_clock::now();
        time_batch += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    std::cout << "batch: " << time_batch << "ns" << std::endl;

    uint64_t time_batch_one = 0;
    for (int i = 99999; i >= 0; i--) {
        Node node;
        auto pos1 = rand() % 100000;
        uint64_t addr = rdma->mem().addr + pos1 * sizeof(Node);
        // auto t1 = std::chrono::high_resolution_clock::now();
        rdma->read_batch_add((void*)addr, &node, sizeof(Node));
        rdma->batch_commit();
        // auto t2 = std::chrono::high_resolution_clock::now();
        // time_batch_one += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    for (int i = 99999; i >= 0; i--) {
        Node node;
        auto pos1 = rand() % 100000;
        uint64_t addr = rdma->mem().addr + pos1 * sizeof(Node);
        auto t1 = std::chrono::high_resolution_clock::now();
        rdma->read_batch_add((void*)addr, &node, sizeof(Node));
        rdma->batch_commit();
        auto t2 = std::chrono::high_resolution_clock::now();
        time_batch_one += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    std::cout << "batch one: " << time_batch_one << "ns" << std::endl;

    sleep(1);

    uint64_t time_no_batch = 0;
    for (int i = 99999; i >= 0; i--) {
        Node node;
        auto pos1 = rand() % 100000;
        uint64_t addr = rdma->mem().addr + pos1 * sizeof(Node);
        // auto t1 = std::chrono::high_resolution_clock::now();
        rdma->read((void*)addr, &node, sizeof(Node));
        // auto t2 = std::chrono::high_resolution_clock::now();
        // time_no_batch += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    for (int i = 99999; i >= 0; i--) {
        Node node;
        auto pos1 = rand() % 100000;
        uint64_t addr = rdma->mem().addr + pos1 * sizeof(Node);
        auto t1 = std::chrono::high_resolution_clock::now();
        rdma->read((void*)addr, &node, sizeof(Node));
        auto t2 = std::chrono::high_resolution_clock::now();
        time_no_batch += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    std::cout << "no batch: " << time_no_batch << "ns" << std::endl;

    uint64_t time_write_read = 0;
    for (int i = 99999; i >= 0; i -= 2) {
        Node node1, node2;
        auto pos1 = rand() % 100000;
        auto pos2 = rand() % 100000;
        uint64_t addr1 = rdma->mem().addr + pos1 * sizeof(Node);
        uint64_t addr2 = rdma->mem().addr + pos2 * sizeof(Node);
        for (int j = 0; j < 128; j++) {
            node1.buf[j] = ((uint64_t)rand() << 32) + rand();
        }
        // auto t1 = std::chrono::high_resolution_clock::now();
        rdma->write(&node1, (void*)addr1, (uint32_t)sizeof(Node));
        rdma->read((void*)addr2, &node2, (uint32_t)sizeof(Node));
        // auto t2 = std::chrono::high_resolution_clock::now();
        // time_write_read += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    for (int i = 99999; i >= 0; i -= 2) {
        Node node1, node2;
        auto pos1 = rand() % 100000;
        auto pos2 = rand() % 100000;
        uint64_t addr1 = rdma->mem().addr + pos1 * sizeof(Node);
        uint64_t addr2 = rdma->mem().addr + pos2 * sizeof(Node);
        for (int j = 0; j < 128; j++) {
            node1.buf[j] = ((uint64_t)rand() << 32) + rand();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        rdma->write(&node1, (void*)addr1, (uint32_t)sizeof(Node));
        rdma->read((void*)addr2, &node2, (uint32_t)sizeof(Node));
        auto t2 = std::chrono::high_resolution_clock::now();
        time_write_read += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    std::cout << "write read no batch: " << time_write_read << "ns" << std::endl;

    uint64_t time_write_read_batch = 0;
    for (int i = 99999; i >= 0; i -= 2) {
        Node node1, node2;
        auto pos1 = rand() % 100000;
        auto pos2 = rand() % 100000;
        uint64_t addr1 = rdma->mem().addr + pos1 * sizeof(Node);
        uint64_t addr2 = rdma->mem().addr + pos2 * sizeof(Node);
        for (int j = 0; j < 128; j++) {
            node1.buf[j] = ((uint64_t)rand() << 32) + rand();
        }
        // auto t1 = std::chrono::high_resolution_clock::now();
        rdma->write_batch_add(&node1, (void*)addr1, (uint32_t)sizeof(Node));
        rdma->read_batch_add((void*)addr2, &node2, (uint32_t)sizeof(Node));
        rdma->batch_commit();
        // auto t2 = std::chrono::high_resolution_clock::now();
        // time_write_read_batch += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    for (int i = 99999; i >= 0; i -= 2) {
        Node node1, node2;
        auto pos1 = rand() % 100000;
        auto pos2 = rand() % 100000;
        uint64_t addr1 = rdma->mem().addr + pos1 * sizeof(Node);
        uint64_t addr2 = rdma->mem().addr + pos2 * sizeof(Node);
        for (int j = 0; j < 128; j++) {
            node1.buf[j] = ((uint64_t)rand() << 32) + rand();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        rdma->write_batch_add(&node1, (void*)addr1, (uint32_t)sizeof(Node));
        rdma->read_batch_add((void*)addr2, &node2, (uint32_t)sizeof(Node));
        rdma->batch_commit();
        auto t2 = std::chrono::high_resolution_clock::now();
        time_write_read_batch += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    std::cout << "write read batch: " << time_write_read_batch << "ns" << std::endl;

    return 0;
}
