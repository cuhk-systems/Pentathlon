#ifndef PENTATHLON_BENCHMARKS_NEW_YCSB_GENERATORS_ZIPF_H_
#define PENTATHLON_BENCHMARKS_NEW_YCSB_GENERATORS_ZIPF_H_

#include <stdint.h>

#include "xoshiro256p.h"

struct pth_zipf {
    uint64_t n;
    double theta;
    double alpha;

    double zeta_two_theta;
    double zeta_n_theta;

    double half_pow_theta;
    double eta;

    struct pth_xoshiro256p uniform;
};

int pth_zipf_init(struct pth_zipf *gen, uint64_t n, double theta,
                  struct pth_xoshiro256p *uniform_p);
uint64_t pth_zipf_next(struct pth_zipf *gen);

#endif
