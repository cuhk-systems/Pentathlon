#pragma once

#include <iostream>
#include <vector>
#include <list>
#include <mutex>
#include <random>
#include <atomic>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <chrono>
#include "lock.h"

template<typename Key, typename Value, typename Hash = std::hash<Key>>
class ThreadSafeHashTable {
private:
    struct alignas(64) Bucket {
        std::list<std::pair<Key, Value>> data;
        Lock mutex;
        
        // 查找元素
        std::optional<Value> find(const Key& key) {
            // std::shared_lock lock(mutex);
            mutex.get_lock();
            for (const auto& pair : data) {
                if (pair.first == key) {
                    auto res = pair.second;
                    mutex.release_lock();
                    return res;
                }
            }
            mutex.release_lock();
            return std::nullopt;
        }
        
        // 插入元素
        bool insert(const Key& key, const Value& value) {
            // std::unique_lock lock(mutex);
            mutex.get_lock();
            for (auto it = data.begin(); it != data.end(); ++it) {
                if (it->first == key) {
                    data.erase(it);
                    break;
                }
            }
            data.emplace_back(key, value);
            mutex.release_lock();
            return true;
        }
        
        // 删除元素
        bool erase(const Key& key) {
            // std::unique_lock lock(mutex);
            mutex.get_lock();
            for (auto it = data.begin(); it != data.end(); ++it) {
                if (it->first == key) {
                    data.erase(it);
                    mutex.release_lock();
                    return true;
                }
            }
            mutex.release_lock();
            return false;
        }
        
        // 获取随机元素
        std::optional<std::pair<Key, Value>> get_random() {
            // std::shared_lock lock(mutex);
            mutex.get_lock();
            if (data.empty()) {
                mutex.release_lock();
                return std::nullopt;
            }
            
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
            
            auto it = data.begin();
            std::advance(it, dist(gen));
            auto result = *it;
            mutex.release_lock();
            return result;
        }
        
        // 获取桶大小
        size_t size() {
            // std::shared_lock lock(mutex);
            mutex.get_lock();
            auto res = data.size();
            mutex.release_lock();
            return res;
        }
    };
    
    Bucket* buckets;
    int bucket_count;
    Hash hasher;
    std::atomic<size_t> element_count{0};
    
    // 获取桶索引
    size_t get_bucket_index(const Key& key) const {
        return hasher(key) % bucket_count;
    }

public:
    // 构造函数
    ThreadSafeHashTable(size_t _bucket_count = 61) : bucket_count(_bucket_count) {
        buckets = new Bucket[bucket_count];
        // 使用质数作为桶数量以减少哈希冲突
    }
    
    // 插入键值对
    bool insert(const Key& key, const Value& value) {
        size_t index = get_bucket_index(key);
        bool inserted = buckets[index].insert(key, value);
        if (inserted) {
            element_count.fetch_add(1, std::memory_order_relaxed);
        }
        return inserted;
    }
    
    // 查找元素
    std::optional<Value> find(const Key& key) const {
        size_t index = get_bucket_index(key);
        return buckets[index].find(key);
    }
    
    // 删除元素
    bool erase(const Key& key) {
        size_t index = get_bucket_index(key);
        bool erased = buckets[index].erase(key);
        if (erased) {
            element_count.fetch_sub(1, std::memory_order_relaxed);
        }
        return erased;
    }
    
    // 随机删除一个元素
    std::optional<std::pair<int, Value>> random_erase() {
        if (empty()) {
            return std::nullopt;
        }
        
        // 尝试从随机桶中删除元素
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, bucket_count - 1);
        
        // 最多尝试所有桶一次
        for (size_t i = 0; i < bucket_count; i++) {
            size_t index = dist(gen);
            auto& bucket = buckets[index];
            
            // std::unique_lock lock(bucket.mutex);
            bucket.mutex.get_lock();
            if (!bucket.data.empty()) {
                auto it = bucket.data.begin();
                auto result = *it;
                bucket.data.erase(it);
                element_count.fetch_sub(1, std::memory_order_relaxed);
                bucket.mutex.release_lock();
                return std::make_pair(index, result.second);
            }
            bucket.mutex.release_lock();
        }
        
        return std::nullopt;
    }

    void release_lock(int index) {
        buckets[index].mutex.release_lock();
    }
    
    // 获取大小
    size_t size() const {
        return element_count.load(std::memory_order_relaxed);
    }
    
    // 检查是否为空
    bool empty() const {
        return size() == 0;
    }
    
    // 获取负载因子
    double load_factor() const {
        return static_cast<double>(size()) / bucket_count;
    }
    
    // 清空哈希表
    void clear() {
        for (int i = 0; i < bucket_count; i++) {
            auto& bucket = buckets[i];
            // std::unique_lock lock(bucket.mutex);
            bucket.mutex.get_lock();
            bucket.data.clear();
            bucket.mutex.release_lock();
        }
        element_count.store(0, std::memory_order_relaxed);
    }
};