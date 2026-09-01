#ifndef RUNTIME_COMPUTE_SRC_EPOCHMANAGER_HPP_
#define RUNTIME_COMPUTE_SRC_EPOCHMANAGER_HPP_

#include <atomic>
#include <cstdint>
#include <functional>

class EpochManager {
private:
    struct ThreadEpoch {
        std::atomic<uint64_t> local_epoch{0};
        std::atomic<bool> active{false};
        std::vector<std::function<void()>> retired_objects[3];
    };

    alignas(64) std::atomic<uint64_t> global_epoch{0};
    alignas(64) std::vector<ThreadEpoch*> thread_epochs;
    std::atomic<bool> shutdown{false};
    
    // 线程本地存储
    static inline thread_local ThreadEpoch* local_epoch = nullptr;
    static inline thread_local uint32_t thread_id = 0;

public:
    EpochManager() {
        // 预留空间避免重哈希
        thread_epochs.reserve(64);
    }

    ~EpochManager() {
        shutdown.store(true, std::memory_order_release);
        
        // 清理所有线程数据
        for (auto* te : thread_epochs) {
            if (te) {
                delete te;
            }
        }
    }

    // 注册当前线程到epoch管理器
    void register_thread() {
        if (!local_epoch) {
            auto* te = new ThreadEpoch();
            te->active.store(true, std::memory_order_release);
            te->local_epoch.store(global_epoch.load(std::memory_order_acquire), 
                                 std::memory_order_release);
            
            thread_id = thread_epochs.size();
            thread_epochs.push_back(te);
            local_epoch = te;
        }
    }

    // 注销当前线程
    void unregister_thread() {
        if (local_epoch) {
            // 尝试回收所有剩余对象
            try_reclaim();
            
            local_epoch->active.store(false, std::memory_order_release);
            local_epoch = nullptr;
        }
    }

    // 进入临界区
    uint64_t enter_critical_section() {
        if (!local_epoch) {
            register_thread();
        }
        
        // 读取当前全局epoch
        uint64_t epoch = global_epoch.load(std::memory_order_acquire);
        local_epoch->local_epoch.store(epoch, std::memory_order_release);
        return epoch;
    }

    // 退出临界区
    void exit_critical_section() {
        if (local_epoch) {
            local_epoch->local_epoch.store(-1, std::memory_order_release);
        }
    }

    // 退役对象并在安全时回收，传入删除函数
    // template<typename T>
    void retire(std::function<void()> deleter) {
        if (!local_epoch || shutdown.load(std::memory_order_acquire)) {
            deleter();
            return;
        }

        uint32_t epoch_index = global_epoch.load(std::memory_order_acquire) % 3;

        // 将删除操作包装为lambda
        local_epoch->retired_objects[epoch_index].emplace_back(deleter);
        // 尝试回收
        try_reclaim();
    }

private:
    // 尝试回收可安全删除的对象
    void try_reclaim() {
        if (!local_epoch || shutdown.load(std::memory_order_acquire)) {
            return;
        }

        uint64_t ge = global_epoch.load(std::memory_order_acquire);

        for (auto* te : thread_epochs) {
            if (te && te->active.load(std::memory_order_acquire)) {
                uint64_t le = te->local_epoch.load(std::memory_order_acquire);
                if (le != -1 && le != ge) {
                    return;
                }
            }
        }

        // 推进全局epoch
        uint64_t new_epoch = (ge + 1) % 3;
        if (global_epoch.compare_exchange_weak(ge, new_epoch, 
                                             std::memory_order_acq_rel)) {
                
            // 回收两个epoch前的对象
            uint32_t reclaim_epoch_index = (new_epoch + 1) % 3;
            reclaim_epoch(reclaim_epoch_index);
        }
    }

    // 回收特定epoch的对象
    void reclaim_epoch(uint32_t epoch_index) {
        for (auto* te : thread_epochs) {
            if (te) {
                auto& objects = te->retired_objects[epoch_index];
                for (auto& deleter : objects) {
                    deleter();
                }
                objects.clear();
            }
        }
    }
};

#endif  // RUNTIME_COMPUTE_SRC_EPOCHMANAGER_HPP_