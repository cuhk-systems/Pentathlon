#include <stdint.h>
#include <stdlib.h>

int test_runtime_hooks(void) {
  int *p = (int *)malloc(sizeof(int));
  *p = 7;

  int *alias = p;
  int value = *alias;

  uint64_t x = 0;
  uint64_t old = __atomic_load_n(&x, __ATOMIC_SEQ_CST);
  __atomic_store_n(&x, old + 1, __ATOMIC_SEQ_CST);
  __atomic_fetch_add(&x, 2, __ATOMIC_SEQ_CST);
  uint64_t expected = 3;
  __atomic_compare_exchange_n(&x, &expected, 4, 0, __ATOMIC_SEQ_CST,
                              __ATOMIC_SEQ_CST);

  free(p);
  return value + (int)x;
}
