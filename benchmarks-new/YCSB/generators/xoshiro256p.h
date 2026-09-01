#ifndef PENTATHLON_BENCHMARKS_NEW_YCSB_GENERATORS_XOSHIRO256P_H_
#define PENTATHLON_BENCHMARKS_NEW_YCSB_GENERATORS_XOSHIRO256P_H_

#include <stdint.h>

struct pth_xoshiro256p {
    uint64_t s[4];
};

void pth_xoshiro256p_init(struct pth_xoshiro256p *gen, uint64_t seed);
uint64_t pth_xoshiro256p_next(struct pth_xoshiro256p *gen);
double pth_xoshiro256p_next_f64(struct pth_xoshiro256p *gen);
void pth_xoshiro256p_jump(struct pth_xoshiro256p *gen);

#endif
