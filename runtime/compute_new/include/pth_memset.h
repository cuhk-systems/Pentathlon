#ifndef PTH_MEMSET_H
#define PTH_MEMSET_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

void *pth_memset(void *s, int c, size_t n);
void *pth_memcpy(void *__restrict dest, const void *__restrict src, size_t n);
void *pth_memmove(void *__restrict dest, const void *__restrict src, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // PTH_MEMSET_H
