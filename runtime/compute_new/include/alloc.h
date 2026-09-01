#ifndef COMPUTE_ALLOC_H
#define COMPUTE_ALLOC_H

#include <stddef.h>

#include "addr.h"

#ifdef __cplusplus
extern "C" {
#endif

void* disaggAlloc(size_t size);
void disaggFree(void* gaddr);

#ifdef __cplusplus
}
#endif

#endif
