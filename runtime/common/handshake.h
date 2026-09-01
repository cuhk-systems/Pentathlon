// Information exchanged at creation of a connection.

#ifndef _COMMON_HANDSHAKE_H_
#define _COMMON_HANDSHAKE_H_

#include <stdint.h>

// Can't think of anything that compute side need to pass to memory side for now.
struct compute_info {
  uint8_t _reserved[16];
};

struct memory_info {
  uint64_t addr, page_size;
  uint32_t rkey, page_count;
};

// Remote memory pool range [info.addr, mem_upper_bound(info))
static inline uint64_t mem_upper_bound(struct memory_info info) {
  return info.addr + info.page_size * info.page_count;
}

#endif
