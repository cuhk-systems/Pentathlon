#include "zipf.h"

#include <errno.h>
#include <math.h>
#include <time.h>

static double zeta(uint64_t n, double theta) {
    double z = 0.0;
    for (uint64_t i = 1; i <= n; ++i) z += pow(i, -theta);
    return z;
}

int pth_zipf_init(struct pth_zipf *gen, uint64_t n, double theta,
                  struct pth_xoshiro256p *uniform_p) {
    if (n < 2) return -EINVAL;

    if (theta <= 0.0 || theta >= 1.0) return -EINVAL;

    gen->n = n;
    gen->theta = theta;
    gen->alpha = 1.0 / (1.0 - theta);

    gen->zeta_two_theta = zeta(2, theta);
    gen->zeta_n_theta = zeta(n, theta);

    gen->half_pow_theta = pow(0.5, gen->theta);
    gen->eta = (1.0 - pow(2.0 / n, 1.0 - theta)) / (1.0 - gen->zeta_two_theta / gen->zeta_n_theta);

    if (uniform_p)
        gen->uniform = *uniform_p;
    else
        pth_xoshiro256p_init(&gen->uniform, time(NULL));

    return 0;
}

uint64_t pth_zipf_next(struct pth_zipf *gen) {
    double u = pth_xoshiro256p_next_f64(&gen->uniform);
    double uzn = u * gen->zeta_n_theta;

    if (uzn < 1.0) return 1;

    if (uzn < 1.0 + gen->half_pow_theta) return 2;

    long result = gen->n * pow(gen->eta * u - gen->eta + 1.0, gen->alpha);

    if (result < 1) result = 1;

    if (result > gen->n) result = gen->n;

    return result;
}
