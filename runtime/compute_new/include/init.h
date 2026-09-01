#ifndef COMPUTE_INIT_H
#define COMPUTE_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

// Ensures the static library is used and let the global state constructor to run.
// Intended for test only.
void __ensureUsed();

#ifdef __cplusplus
}
#endif

#endif  // COMPUTE_INIT_H
