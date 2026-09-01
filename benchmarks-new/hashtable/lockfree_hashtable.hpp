#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace pth_hashtable {

class LockFreeHashTable {
  static constexpr uintptr_t kMark = 1;
  static constexpr size_t kSegmentBits = 7;
  static constexpr size_t kBucketsPerSegment = 1ULL << kSegmentBits;
  static constexpr size_t kSegmentMask = kBucketsPerSegment - 1;
  static constexpr size_t kDirectoryBits = 11;
  static constexpr size_t kSegmentsPerDirectory = 1ULL << kDirectoryBits;
  static constexpr size_t kDirectoryMask = kSegmentsPerDirectory - 1;
  static constexpr size_t kDirectoryCount = 128;
  static constexpr size_t kSegmentCount = kDirectoryCount * kSegmentsPerDirectory;
  static constexpr size_t kBucketCount = kSegmentCount * kBucketsPerSegment;
  static constexpr size_t kMaxAllocation = 1ULL << 14;
  static constexpr size_t kNodeSize = 1ULL << 10;

  struct Node {
    int key;
    std::atomic<int> value;
    std::atomic<uintptr_t> next;
    std::atomic<Node *> allocated_next;
    std::byte padding[kNodeSize - sizeof(int) - sizeof(std::atomic<int>) -
                      sizeof(std::atomic<uintptr_t>) - sizeof(std::atomic<Node *>)];

    Node(int key, int value) : key(key), value(value), next(0), allocated_next(nullptr) {}
  };

  struct Segment {
    std::atomic<uintptr_t> buckets[kBucketsPerSegment];

    Segment() {
      for (size_t i = 0; i < kBucketsPerSegment; ++i) {
        buckets[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  struct Directory {
    std::atomic<Segment *> segments[kSegmentsPerDirectory];

    Directory() {
      for (size_t i = 0; i < kSegmentsPerDirectory; ++i) {
        segments[i].store(nullptr, std::memory_order_relaxed);
      }
    }
  };

  static_assert(sizeof(Node) == kNodeSize, "node must occupy exactly 1 KiB");
  static_assert(sizeof(Node) <= kMaxAllocation, "node allocation is too large");
  static_assert(sizeof(Segment) == kNodeSize, "bucket segment must occupy exactly 1 KiB");
  static_assert(sizeof(Directory) <= kMaxAllocation, "directory allocation is too large");
  static_assert(alignof(Node) >= 2, "node pointers must have a free low mark bit");

  std::atomic<Directory *> directories_[kDirectoryCount];
  std::atomic<Node *> allocated_nodes_;
  std::atomic<uint64_t> item_count_;

  static_assert(kDirectoryCount * sizeof(std::atomic<Directory *>) +
                        sizeof(std::atomic<Node *>) + sizeof(std::atomic<uint64_t>) <=
                    kMaxAllocation,
                "hashtable object allocation is too large");

  static Node *ptr(uintptr_t ref) { return reinterpret_cast<Node *>(ref & ~kMark); }

  static uintptr_t ref(Node *node) { return reinterpret_cast<uintptr_t>(node); }

  static bool marked(uintptr_t ref) { return (ref & kMark) != 0; }

  static uint64_t hash_key(int key) {
    uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(key));
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
  }

  Directory *get_directory(size_t directory_index, bool create) {
    Directory *directory = directories_[directory_index].load(std::memory_order_acquire);
    if (directory != nullptr || !create) return directory;

    Directory *candidate = new Directory();
    Directory *expected = nullptr;
    if (directories_[directory_index].compare_exchange_strong(expected, candidate,
                                                              std::memory_order_release,
                                                              std::memory_order_acquire)) {
      return candidate;
    }

    delete candidate;
    return expected;
  }

  Segment *get_segment(size_t segment_index, bool create) {
    Directory *directory = get_directory(segment_index >> kDirectoryBits, create);
    if (directory == nullptr) return nullptr;

    std::atomic<Segment *> &slot = directory->segments[segment_index & kDirectoryMask];
    Segment *segment = slot.load(std::memory_order_acquire);
    if (segment != nullptr || !create) return segment;

    Segment *candidate = new Segment();
    Segment *expected = nullptr;
    if (slot.compare_exchange_strong(expected, candidate, std::memory_order_release,
                                     std::memory_order_acquire)) {
      return candidate;
    }

    delete candidate;
    return expected;
  }

  std::atomic<uintptr_t> *bucket_for(int key, bool create) {
    size_t index = hash_key(key) & (kBucketCount - 1);
    Segment *segment = get_segment(index >> kSegmentBits, create);
    if (segment == nullptr) return nullptr;
    return &segment->buckets[index & kSegmentMask];
  }

  void remember(Node *node) {
    Node *old_head = allocated_nodes_.load(std::memory_order_acquire);
    do {
      node->allocated_next.store(old_head, std::memory_order_relaxed);
    } while (!allocated_nodes_.compare_exchange_weak(old_head, node, std::memory_order_release,
                                                     std::memory_order_acquire));
  }

  void cleanup_bucket(std::atomic<uintptr_t> *bucket) {
  restart:
    std::atomic<uintptr_t> *link = bucket;
    uintptr_t curr_ref = link->load(std::memory_order_acquire);
    Node *curr = ptr(curr_ref);

    while (curr != nullptr) {
      uintptr_t next_ref = curr->next.load(std::memory_order_acquire);
      Node *next = ptr(next_ref);

      if (marked(next_ref)) {
        uintptr_t expected = ref(curr);
        if (!link->compare_exchange_strong(expected, ref(next), std::memory_order_acq_rel,
                                           std::memory_order_acquire)) {
          goto restart;
        }
        curr = next;
        continue;
      }

      link = &curr->next;
      curr = next;
    }
  }

 public:
  LockFreeHashTable() : allocated_nodes_(nullptr), item_count_(0) {
    for (size_t i = 0; i < kDirectoryCount; ++i) {
      directories_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  ~LockFreeHashTable() {
    Node *node = allocated_nodes_.load(std::memory_order_acquire);
    while (node != nullptr) {
      Node *next = node->allocated_next.load(std::memory_order_relaxed);
      delete node;
      node = next;
    }

    for (size_t i = 0; i < kDirectoryCount; ++i) {
      Directory *directory = directories_[i].load(std::memory_order_relaxed);
      if (directory == nullptr) continue;

      for (size_t j = 0; j < kSegmentsPerDirectory; ++j) {
        delete directory->segments[j].load(std::memory_order_relaxed);
      }
      delete directory;
    }
  }

  LockFreeHashTable(const LockFreeHashTable &) = delete;
  LockFreeHashTable &operator=(const LockFreeHashTable &) = delete;

  bool get(int key, int *value) {
    std::atomic<uintptr_t> *bucket = bucket_for(key, false);
    if (bucket == nullptr) return false;

    Node *curr = ptr(bucket->load(std::memory_order_acquire));
    while (curr != nullptr) {
      uintptr_t next_ref = curr->next.load(std::memory_order_acquire);
      if (!marked(next_ref) && curr->key == key) {
        if (value != nullptr) *value = curr->value.load(std::memory_order_acquire);
        return true;
      }
      curr = ptr(next_ref);
    }

    return false;
  }

  bool put(int key, int value) {
    std::atomic<uintptr_t> *bucket = bucket_for(key, true);
    if (bucket == nullptr) return false;

    for (;;) {
      uintptr_t head = bucket->load(std::memory_order_acquire);
      Node *curr = ptr(head);
      while (curr != nullptr) {
        uintptr_t next_ref = curr->next.load(std::memory_order_acquire);
        if (!marked(next_ref) && curr->key == key) {
          curr->value.store(value, std::memory_order_release);
          return false;
        }
        curr = ptr(next_ref);
      }

      Node *node = new Node(key, value);
      node->next.store(head, std::memory_order_relaxed);

      if (bucket->compare_exchange_strong(head, ref(node), std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
        remember(node);
        item_count_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }

      delete node;
      cleanup_bucket(bucket);
    }
  }

  bool erase(int key) {
    std::atomic<uintptr_t> *bucket = bucket_for(key, false);
    if (bucket == nullptr) return false;

    for (;;) {
      bool retry = false;
      Node *curr = ptr(bucket->load(std::memory_order_acquire));
      while (curr != nullptr) {
        uintptr_t next_ref = curr->next.load(std::memory_order_acquire);
        Node *next = ptr(next_ref);

        if (!marked(next_ref) && curr->key == key) {
          uintptr_t desired = next_ref | kMark;
          if (curr->next.compare_exchange_strong(next_ref, desired, std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
            item_count_.fetch_sub(1, std::memory_order_relaxed);
            cleanup_bucket(bucket);
            return true;
          }
          cleanup_bucket(bucket);
          retry = true;
          break;
        }

        curr = next;
      }

      if (retry) continue;
      return false;
    }
  }

  uint64_t size() const { return item_count_.load(std::memory_order_relaxed); }

  static constexpr size_t bucket_count() { return kBucketCount; }
};

}  // namespace pth_hashtable
