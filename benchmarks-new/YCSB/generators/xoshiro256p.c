#include "xoshiro256p.h"

#include "splitmix64.h"

void pth_xoshiro256p_init(struct pth_xoshiro256p *gen, uint64_t seed) {
    for (int i = 0; i < 4; ++i) gen->s[i] = pth_splitmix64_next(&seed);
}

static uint64_t rotate_left(uint64_t val, int shift) {
    return (val << shift) | (val >> (64 - shift));
}

uint64_t pth_xoshiro256p_next(struct pth_xoshiro256p *gen) {
    uint64_t *s = gen->s;

    uint64_t result = s[0] + s[3];
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotate_left(s[3], 45);

    return result;
}

double pth_xoshiro256p_next_f64(struct pth_xoshiro256p *gen) {
    uint64_t full = pth_xoshiro256p_next(gen);
    return (full >> 11) * 0x1.0p-53;
}

void pth_xoshiro256p_jump(struct pth_xoshiro256p *gen) {
    const uint64_t JUMP[] = {
        0x180EC6D33CFD0ABA,
        0xD5A61266F0C9392C,
        0xA9582618E03FC9AA,
        0x39ABDC4529B1661C,
    };

    uint64_t *s = gen->s;

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 64; ++j) {
            if (JUMP[i] & (1ULL << j)) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }

            pth_xoshiro256p_next(gen);
        }
    }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}
