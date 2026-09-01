#ifndef _COMMON_PARSE_H_
#define _COMMON_PARSE_H_

#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include "try.h"

static inline int parse_port(const char* port_str) {
  if (!port_str || port_str[0] == '\0') return -(errno = EINVAL);
  char* endptr;
  long result = strtol(port_str, &endptr, 10);
  if (*endptr != '\0') return -(errno = EINVAL);
  if (result < 0 || result > 65535) return -(errno = ERANGE);
  return (int)result;
}

// parse ip:port
static inline int parse_addr(char* addr_str, struct sockaddr_storage* addr, int default_port) {
  if (addr_str[0] == '[') {
    // IPv4 or v6, with square brackets surrounding IP
    char* close_bracket = strrchr(addr_str, ']');
    if (!close_bracket) {
      fprintf(stderr, "missing closing ']' in address\n");
      return -(errno = EINVAL);
    }
    close_bracket[0] = '\0';

    addr_str += 1;  // it should now only contains IP address
    in_port_t* port_ptr;
    if (strchr(addr_str, ':')) {
      // IPv6
      struct sockaddr_in6* addr_v6 = (typeof(addr_v6))addr;
      addr_v6->sin6_family = AF_INET6;
      port_ptr = &addr_v6->sin6_port;
      char* scope_sep = strrchr(addr_str, '%');
      if (scope_sep) {
        scope_sep[0] = '\0';
        char* scope_str = scope_sep + 1;
        addr_v6->sin6_scope_id = if_nametoindex(scope_str);
        if (addr_v6->sin6_scope_id == 0) {
          fprintf(stderr, "no interface named '%s'\n", scope_str);
          return -(errno = EINVAL);
        }
      }
      try(inet_pton(AF_INET6, addr_str, &addr_v6->sin6_addr), "failed to parse IPv6 address");

    } else {
      // IPv4
      struct sockaddr_in* addr_v4 = (typeof(addr_v4))addr;
      addr_v4->sin_family = AF_INET;
      port_ptr = &addr_v4->sin_port;
      try(inet_pton(AF_INET, addr_str, &addr_v4->sin_addr), "failed to parse IPv4 address");
    }

    switch (close_bracket[1]) {
      case '\0':
        if (default_port < 0) {
          fprintf(stderr, "address missing port\n");
          return -(errno = EINVAL);
        }
        *port_ptr = htons((uint16_t)default_port);
        break;
      case ':':
        *port_ptr = htons((uint16_t)try(parse_port(close_bracket + 2), "failed to parse port"));
        break;
      default:
        fprintf(stderr, "invalid trailing bytes '%s'\n", close_bracket + 1);
        break;
    }

  } else {
    // Only IPv4 is valid without bracket
    struct sockaddr_in* addr_v4 = (typeof(addr_v4))addr;
    addr_v4->sin_family = AF_INET;
    char* port_sep = strrchr(addr_str, ':');
    if (!port_sep) {
      if (default_port < 0) {
        fprintf(stderr, "address missing port\n");
        return -(errno = EINVAL);
      }
      addr_v4->sin_port = htons((uint16_t)default_port);
    } else {
      port_sep[0] = '\0';
      addr_v4->sin_port = htons((uint16_t)try(parse_port(port_sep + 1), "failed to parse port"));
    }
    try(inet_pton(AF_INET, addr_str, &addr_v4->sin_addr), "failed to parse IPv4 address");
  }

  return 0;
}

#endif
