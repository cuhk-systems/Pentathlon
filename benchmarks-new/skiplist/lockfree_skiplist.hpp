#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <time.h>

namespace pth_skiplist {

class LockFreeSkipList {
  static constexpr int kMaxLevel = 32;
  static constexpr uintptr_t kMark = 1;

  struct Node {
    int key;
    std::atomic<int> value;
    int height;
    std::atomic<uintptr_t> next[kMaxLevel];
    std::atomic<Node *> allocated_next;

    Node(int key, int value, int height)
        : key(key), value(value), height(height), allocated_next(nullptr) {
      for (int i = 0; i < kMaxLevel; ++i) {
        next[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  static_assert(alignof(Node) >= 2, "node pointers must have a free low mark bit");

  std::atomic<Node *> head_;
  std::atomic<Node *> allocated_;

  static thread_local uint64_t rng_state_;

  static Node *ptr(uintptr_t ref) {
    return reinterpret_cast<Node *>(ref & ~kMark);
  }

  static uintptr_t ref(Node *node) {
    return reinterpret_cast<uintptr_t>(node);
  }

  static bool marked(uintptr_t ref) {
    return (ref & kMark) != 0;
  }

  static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  static uint64_t next_random() {
    if (rng_state_ == 0) seed_thread();
    uint64_t x = rng_state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state_ = x;
    return x * 0x2545f4914f6cdd1dULL;
  }

  static int random_height() {
    int height = 1;
    uint64_t bits = next_random();
    while ((bits & 1) != 0 && height < kMaxLevel) {
      ++height;
      bits >>= 1;
      if (bits == 0) bits = next_random();
    }
    return height;
  }

  void remember(Node *node) {
    Node *old_head = allocated_.load(std::memory_order_acquire);
    do {
      node->allocated_next.store(old_head, std::memory_order_relaxed);
    } while (!allocated_.compare_exchange_weak(old_head, node, std::memory_order_release,
                                               std::memory_order_acquire));
  }

  bool find_position(int key, Node **preds, Node **succs) {
  retry:
    Node *pred = head_.load(std::memory_order_acquire);

    for (int level = kMaxLevel - 1; level >= 0; --level) {
      uintptr_t curr_ref = pred->next[level].load(std::memory_order_acquire);
      Node *curr = ptr(curr_ref);

      while (curr != nullptr) {
        uintptr_t succ_ref = curr->next[level].load(std::memory_order_acquire);

        while (marked(succ_ref)) {
          Node *succ = ptr(succ_ref);
          uintptr_t expected = ref(curr);
          if (!pred->next[level].compare_exchange_strong(
                  expected, ref(succ), std::memory_order_acq_rel, std::memory_order_acquire)) {
            goto retry;
          }

          curr_ref = pred->next[level].load(std::memory_order_acquire);
          curr = ptr(curr_ref);
          if (curr == nullptr) break;
          succ_ref = curr->next[level].load(std::memory_order_acquire);
        }

        if (curr == nullptr || curr->key >= key) break;
        pred = curr;
        curr = ptr(succ_ref);
      }

      preds[level] = pred;
      succs[level] = curr;
    }

    return succs[0] != nullptr && succs[0]->key == key;
  }

 public:
  LockFreeSkipList() : head_(nullptr), allocated_(nullptr) {
    Node *head = new Node(std::numeric_limits<int>::min(), 0, kMaxLevel);
    remember(head);
    head_.store(head, std::memory_order_release);
  }

  ~LockFreeSkipList() {
    Node *node = allocated_.load(std::memory_order_acquire);
    while (node != nullptr) {
      Node *next = node->allocated_next.load(std::memory_order_relaxed);
      delete node;
      node = next;
    }
  }

  LockFreeSkipList(const LockFreeSkipList &) = delete;
  LockFreeSkipList &operator=(const LockFreeSkipList &) = delete;

  static void seed_thread() {
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t seed = static_cast<uint64_t>(ts.tv_sec) * 1000000007ULL;
    seed ^= static_cast<uint64_t>(ts.tv_nsec);
    seed ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&rng_state_));
    rng_state_ = splitmix64(seed);
    if (rng_state_ == 0) rng_state_ = 0x9e3779b97f4a7c15ULL;
  }

  bool find(int key, int *value) {
    Node *preds[kMaxLevel];
    Node *succs[kMaxLevel];
    if (!find_position(key, preds, succs)) return false;

    Node *node = succs[0];
    if (marked(node->next[0].load(std::memory_order_acquire))) return false;
    if (value != nullptr) *value = node->value.load(std::memory_order_acquire);
    return true;
  }

  bool insert(int key, int value) {
    Node *preds[kMaxLevel];
    Node *succs[kMaxLevel];

    for (;;) {
      if (find_position(key, preds, succs)) {
        Node *node = succs[0];
        if (marked(node->next[0].load(std::memory_order_acquire))) continue;
        node->value.store(value, std::memory_order_release);
        return false;
      }

      int height = random_height();
      Node *node = new Node(key, value, height);
      for (int level = 0; level < height; ++level) {
        node->next[level].store(ref(succs[level]), std::memory_order_relaxed);
      }

      uintptr_t expected = ref(succs[0]);
      if (!preds[0]->next[0].compare_exchange_strong(
              expected, ref(node), std::memory_order_acq_rel, std::memory_order_acquire)) {
        delete node;
        continue;
      }

      remember(node);

      for (int level = 1; level < height; ++level) {
        for (;;) {
          if (marked(node->next[0].load(std::memory_order_acquire))) return true;

          expected = ref(succs[level]);
          if (preds[level]->next[level].compare_exchange_strong(
                  expected, ref(node), std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
          }
          find_position(key, preds, succs);
        }
      }

      return true;
    }
  }

  bool erase(int key) {
    Node *preds[kMaxLevel];
    Node *succs[kMaxLevel];

    for (;;) {
      if (!find_position(key, preds, succs)) return false;

      Node *node = succs[0];
      for (int level = node->height - 1; level >= 1; --level) {
        uintptr_t succ_ref = node->next[level].load(std::memory_order_acquire);
        while (!marked(succ_ref)) {
          uintptr_t desired = succ_ref | kMark;
          node->next[level].compare_exchange_weak(succ_ref, desired, std::memory_order_acq_rel,
                                                  std::memory_order_acquire);
        }
      }

      uintptr_t succ_ref = node->next[0].load(std::memory_order_acquire);
      while (!marked(succ_ref)) {
        uintptr_t desired = succ_ref | kMark;
        if (node->next[0].compare_exchange_strong(succ_ref, desired, std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
          find_position(key, preds, succs);
          return true;
        }
      }

      return false;
    }
  }
};

inline thread_local uint64_t LockFreeSkipList::rng_state_ = 0;

}  // namespace pth_skiplist
