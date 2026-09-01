// A set of macros that helps propagate error using functions' return values.
//
// For compatibility, both `errno` and (negative) return value are set to errno. However, using
// errno is preferred. When returning int, librdmacm sets errno to `errno` only and returns -1,
// while libibverbs seems to use return value. In almost all cases, `errno` is set if function
// returns a pointer.

#ifndef _COMMON_TRY_H_
#define _COMMON_TRY_H_

#include <errno.h>
#include <stdio.h>

// Some macro hack so that `try*` can accept optional error message.
#define _get_macro(_0, _1, _2, name, ...) name
#define _ret(retval, ...) _get_macro(_, ##__VA_ARGS__, _ret_1, _ret_0, )(retval, __VA_ARGS__)
#define _ret_0(retval, _0) return retval
#define _ret_1(retval, _0, msg) \
  ({                            \
    perror(msg);                \
    return retval;              \
  })
#define _cleanup(_0, ...) _get_macro(_, ##__VA_ARGS__, _cleanup_1, _cleanup_0, )(_, __VA_ARGS__)
#define _cleanup_0(_0, _1) goto cleanup
#define _cleanup_1(_0, _1, msg) \
  ({                            \
    perror(msg);                \
    goto cleanup;               \
  })

// Used when errno is stored in `errno` declared in <errno.h>.
#define _try(retdef, retval, stmt, ...)             \
  ({                                                \
    int ret = stmt;                                 \
    if (ret < 0) retdef(retval, _0, ##__VA_ARGS__); \
    ret;                                            \
  })

// Used when errno is stored negatively in return value.
#define _try_e(retdef, retval, stmt, ...) \
  ({                                      \
    int ret = stmt;                       \
    if (ret < 0) {                        \
      errno = -ret;                       \
      retdef(retval, _0, ##__VA_ARGS__);  \
    }                                     \
    ret;                                  \
  })

// Used when return value of statement is a pointer, and errno is stored in `errno`.
#define _try_p(retdef, retval, stmt, ...)        \
  ({                                             \
    void* ret = stmt;                            \
    if (!ret) retdef(retval, _0, ##__VA_ARGS__); \
    ret;                                         \
  })

// try, try_*: parent returns int
#define try(stmt, ...) _try(_ret, -errno, stmt, ##__VA_ARGS__)
#define try_e(stmt, ...) _try_e(_ret, -errno, stmt, ##__VA_ARGS__)
#define try_p(stmt, ...) _try_p(_ret, -errno, stmt, ##__VA_ARGS__)

// try2, try2_*: parent returns pointer
#define try2(stmt, ...) _try(_ret, NULL, stmt, ##__VA_ARGS__)
#define try2_e(stmt, ...) _try_e(_ret, NULL, stmt, ##__VA_ARGS__)
#define try2_p(stmt, ...) _try_p(_ret, NULL, stmt, ##__VA_ARGS__)

// try3, try3_*: parent contains `cleanup` label
#define try3(stmt, ...) _try(_cleanup, _, stmt, ##__VA_ARGS__)
#define try3_e(stmt, ...) _try_e(_cleanup, _, stmt, ##__VA_ARGS__)
#define try3_p(stmt, ...) _try_p(_cleanup, _, stmt, ##__VA_ARGS__)

#endif
