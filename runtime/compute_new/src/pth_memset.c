#include <string.h>

void *pth_memset(void *s, int c, size_t n) { return memset(s, c, n); }
void *pth_memcpy(void *__restrict dest, const void *__restrict src, size_t n) {
    return memcpy(dest, src, n);
}
void *pth_memmove(void *__restrict dest, const void *__restrict src, size_t n) {
    return memmove(dest, src, n);
}
