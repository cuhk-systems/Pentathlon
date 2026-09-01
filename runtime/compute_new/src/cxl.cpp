#include <numa.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "rdma.hpp"

namespace {
constexpr size_t kDefaultCxlPoolSize = size_t{4} << 30;
constexpr size_t kDefaultCxlPageSize = size_t{2} << 20;

size_t parse_size_env(const char* name, size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value) {
        throw std::runtime_error(std::string("invalid size in ") + name + ": " + value);
    }

    size_t multiplier = 1;
    if (*end != '\0') {
        if (end[1] != '\0') {
            throw std::runtime_error(std::string("invalid size suffix in ") + name + ": " + value);
        }
        switch (*end) {
            case 'k':
            case 'K':
                multiplier = size_t{1} << 10;
                break;
            case 'm':
            case 'M':
                multiplier = size_t{1} << 20;
                break;
            case 'g':
            case 'G':
                multiplier = size_t{1} << 30;
                break;
            default:
                throw std::runtime_error(std::string("invalid size suffix in ") + name + ": " + value);
        }
    }

    if (parsed > std::numeric_limits<size_t>::max() / multiplier) {
        throw std::runtime_error(std::string("size overflow in ") + name + ": " + value);
    }
    return static_cast<size_t>(parsed) * multiplier;
}

int parse_int_env(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }

    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid integer in ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}
}  // namespace

void RDMAClient::init_cxl_pool() {
    cxl_node = parse_int_env("PTH_CXL_NODE", 1);
    cxl_pool_size = parse_size_env("PTH_CXL_POOL_SIZE", kDefaultCxlPoolSize);
    size_t page_size = parse_size_env("PTH_CXL_PAGE_SIZE", kDefaultCxlPageSize);
    if (page_size == 0 || cxl_pool_size < page_size) {
        throw std::runtime_error("invalid CXL pool/page size");
    }

    if (numa_available() < 0) {
        throw std::runtime_error("NUMA is not available; cannot allocate CXL pool");
    }
    if (cxl_node > numa_max_node()) {
        throw std::runtime_error("PTH_CXL_NODE exceeds numa_max_node");
    }

    cxl_pool = numa_alloc_onnode(cxl_pool_size, cxl_node);
    if (!cxl_pool) {
        throw std::runtime_error("failed to allocate CXL NUMA pool");
    }

    remote_mem.addr = reinterpret_cast<uint64_t>(cxl_pool);
    remote_mem.rkey = 0;
    remote_mem.page_size = page_size;
    remote_mem.page_count = static_cast<uint32_t>(cxl_pool_size / page_size);

    std::cout << "CXL Init finished: node=" << cxl_node << ", pool=" << cxl_pool
              << ", size=" << cxl_pool_size << " bytes, page_size=" << page_size << std::endl;
}

RDMAClient::~RDMAClient() {
    std::cout << "CXL read: " << read_count << ", CXL write: " << write_count << std::endl;
    if (cxl_pool) {
        numa_free(cxl_pool, cxl_pool_size);
    }
}

void RDMAClient::validate_cxl_range(void* ptr, uint32_t size) const {
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    auto base = reinterpret_cast<uintptr_t>(cxl_pool);
    auto end = base + cxl_pool_size;
    if (addr < base || addr > end || static_cast<size_t>(end - addr) < size) {
        throw std::runtime_error("CXL access outside memory pool");
    }
}

void RDMAClient::read(void* from, void* to, uint32_t size) {
    validate_cxl_range(from, size);
    std::memcpy(to, from, size);
    read_count.fetch_add(1, std::memory_order_relaxed);
}

void RDMAClient::write(void* from, void* to, uint32_t size) {
    validate_cxl_range(to, size);
    std::memcpy(to, from, size);
    write_count.fetch_add(1, std::memory_order_relaxed);
}

void RDMAClient::read_batch_add(void* from, void* to, uint32_t size) {
    if (ThreadBatch::get_instance().count == MAX_BATCH_SIZE) {
        batch_commit();
    }
    auto& batch = ThreadBatch::get_instance();
    auto& entry = batch.get_next();
    entry.from = from;
    entry.to = to;
    entry.size = size;
    entry.is_write = false;
    batch.read_count++;
}

void RDMAClient::write_batch_add(void* from, void* to, uint32_t size) {
    if (ThreadBatch::get_instance().count == MAX_BATCH_SIZE) {
        batch_commit();
    }
    auto& batch = ThreadBatch::get_instance();
    auto& entry = batch.get_next();
    entry.from = from;
    entry.to = to;
    entry.size = size;
    entry.is_write = true;
    batch.write_count++;
}

void RDMAClient::batch_commit() {
    auto& batch = ThreadBatch::get_instance();
    for (size_t i = 0; i < batch.count; ++i) {
        const auto& entry = batch.entries[i];
        if (entry.is_write) {
            validate_cxl_range(entry.to, entry.size);
            std::memcpy(entry.to, entry.from, entry.size);
        } else {
            validate_cxl_range(entry.from, entry.size);
            std::memcpy(entry.to, entry.from, entry.size);
        }
    }

    read_count.fetch_add(batch.read_count, std::memory_order_relaxed);
    write_count.fetch_add(batch.write_count, std::memory_order_relaxed);
    batch.reset();
}
