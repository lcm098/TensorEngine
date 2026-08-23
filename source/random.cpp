#include "../include/random.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Internal PRNG state
// ============================================================================

static int rng_seeded = 0;

static void ensure_rng_seeded(void) {
    if (!rng_seeded) {
        srand((unsigned int) time(NULL));
        rng_seeded = 1;
    }
}

void random_seed(unsigned int seed) {
    srand(seed);
    rng_seeded = 1;
}

// Uniform double in [0.0, 1.0)
static double rand_uniform_01(void) {
    ensure_rng_seeded();
    return (double) rand() / ((double) RAND_MAX + 1.0);
}

// Uniform double in [low, high)
static double rand_uniform_between(double low, double high) {
    return low + rand_uniform_01() * (high - low);
}

// Standard normal via Box-Muller transform
static double rand_standard_normal(void) {
    double u1 = rand_uniform_01();
    double u2 = rand_uniform_01();

    if (u1 < 1e-12) {
        u1 = 1e-12;
    }

    double mag = sqrt(-2.0 * log(u1));
    return mag * cos(2.0 * M_PI * u2);
}

// Uniform random integer in [0, bound) using rejection sampling to avoid
// modulo bias from rand()
static int rand_int_below(int bound) {
    if (bound <= 0) return 0;

    int limit = RAND_MAX - (RAND_MAX % bound);
    int r;
    do {
        r = rand();
    } while (r >= limit);

    return r % bound;
}

// ============================================================================
// Shape / allocation helpers
// ============================================================================

static size_t shape_total(int shape[], size_t *out_ndim) {
    size_t ndim = calculate_ndim(shape);
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        total *= (size_t) shape[i];
    }
    if (out_ndim) *out_ndim = ndim;
    return total;
}

static TensorEngine* build_tensor_from_host(f64 *host_data, int shape[], size_t ndim, bool __GPU__) {
    int *shape_terminated = (int*) malloc(sizeof(int) * (ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "random: memory allocation failed for shape\n");
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++) {
        shape_terminated[i] = shape[i];
    }
    shape_terminated[ndim] = N;

    TensorEngine *result = tensor(host_data, shape_terminated, __GPU__);
    free(shape_terminated);
    return result;
}

// ============================================================================
// Core distribution generators
// ============================================================================

TensorEngine* rand_f64(f64 low, f64 high, int shape[], bool __GPU__) {
    size_t ndim;
    size_t total = shape_total(shape, &ndim);

    if (ndim == 0) {
        fprintf(stderr, "rand_f64: shape must have at least 1 dimension\n");
        return NULL;
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (total + 1));
    if (!host_data) {
        fprintf(stderr, "rand_f64: memory allocation failed\n");
        return NULL;
    }

    for (size_t i = 0; i < total; i++) {
        host_data[i] = rand_uniform_between(low, high);
    }
    host_data[total] = E;

    TensorEngine *result = build_tensor_from_host(host_data, shape, ndim, __GPU__);
    free(host_data);
    return result;
}

TensorEngine* random_uniform(int shape[], f64 low, f64 high, bool __GPU__) {
    return rand_f64(low, high, shape, __GPU__);
}

TensorEngine* random_normal(int shape[], f64 mean, f64 stddev, bool __GPU__) {
    size_t ndim;
    size_t total = shape_total(shape, &ndim);

    if (ndim == 0) {
        fprintf(stderr, "random_normal: shape must have at least 1 dimension\n");
        return NULL;
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (total + 1));
    if (!host_data) {
        fprintf(stderr, "random_normal: memory allocation failed\n");
        return NULL;
    }

    for (size_t i = 0; i < total; i++) {
        host_data[i] = mean + rand_standard_normal() * stddev;
    }
    host_data[total] = E;

    TensorEngine *result = build_tensor_from_host(host_data, shape, ndim, __GPU__);
    free(host_data);
    return result;
}

TensorEngine* random_randn(int shape[], bool __GPU__) {
    return random_normal(shape, 0.0, 1.0, __GPU__);
}

// Binomial(n_trials, p): sum of n_trials independent Bernoulli(p) draws.
// Simple and numerically robust for reasonable n_trials; not optimized for
// very large n_trials (where a normal approximation would be faster).
TensorEngine* random_binomial(int shape[], int n_trials, f64 p, bool __GPU__) {
    if (n_trials < 0) {
        fprintf(stderr, "random_binomial: n_trials must be non-negative\n");
        return NULL;
    }
    if (p < 0.0 || p > 1.0) {
        fprintf(stderr, "random_binomial: p must be in [0, 1]\n");
        return NULL;
    }

    size_t ndim;
    size_t total = shape_total(shape, &ndim);

    if (ndim == 0) {
        fprintf(stderr, "random_binomial: shape must have at least 1 dimension\n");
        return NULL;
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (total + 1));
    if (!host_data) {
        fprintf(stderr, "random_binomial: memory allocation failed\n");
        return NULL;
    }

    for (size_t i = 0; i < total; i++) {
        int successes = 0;
        for (int trial = 0; trial < n_trials; trial++) {
            if (rand_uniform_01() < p) {
                successes++;
            }
        }
        host_data[i] = (f64) successes;
    }
    host_data[total] = E;

    TensorEngine *result = build_tensor_from_host(host_data, shape, ndim, __GPU__);
    free(host_data);
    return result;
}

// Poisson(lambda) via Knuth's algorithm (product-of-uniforms method).
// Suitable for small-to-moderate lambda; for very large lambda a normal
// approximation would be more efficient, but this stays exact.
static int rand_poisson_knuth(double lambda) {
    double L = exp(-lambda);
    int k = 0;
    double p = 1.0;

    do {
        k++;
        p *= rand_uniform_01();
    } while (p > L);

    return k - 1;
}

TensorEngine* random_poisson(int shape[], f64 lambda, bool __GPU__) {
    if (lambda < 0.0) {
        fprintf(stderr, "random_poisson: lambda must be non-negative\n");
        return NULL;
    }

    size_t ndim;
    size_t total = shape_total(shape, &ndim);

    if (ndim == 0) {
        fprintf(stderr, "random_poisson: shape must have at least 1 dimension\n");
        return NULL;
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (total + 1));
    if (!host_data) {
        fprintf(stderr, "random_poisson: memory allocation failed\n");
        return NULL;
    }

    for (size_t i = 0; i < total; i++) {
        host_data[i] = (f64) rand_poisson_knuth(lambda);
    }
    host_data[total] = E;

    TensorEngine *result = build_tensor_from_host(host_data, shape, ndim, __GPU__);
    free(host_data);
    return result;
}

// ============================================================================
// Shuffle / choice
// ============================================================================

// Fisher-Yates shuffle, operating on t->tensor as a flat sequence, in place.
// GPU tensors are copied to host, shuffled, then copied back.
void random_shuffle(TensorEngine *t) {
    if (t == NULL) {
        fprintf(stderr, "random_shuffle: null tensor argument\n");
        return;
    }
    if (t->size < 2) {
        return; // nothing to shuffle
    }

    f64 *work = NULL;

    if (t->__GPU__) {
        work = (f64*) malloc(sizeof(f64) * t->size);
        if (!work) {
            fprintf(stderr, "random_shuffle: memory allocation failed\n");
            return;
        }
        copyDeviceToHost(work, t->tensor, t->size);
    } else {
        work = t->tensor;
    }

    for (size_t i = t->size - 1; i > 0; i--) {
        size_t j = (size_t) rand_int_below((int) (i + 1));
        f64 tmp = work[i];
        work[i] = work[j];
        work[j] = tmp;
    }

    if (t->__GPU__) {
        copyHostToDevice(t->tensor, work, t->size);
        free(work);
    }
    // if CPU, work == t->tensor, already modified in place
}

// Randomly selects n elements from src (flattened), with or without
// replacement, returning a new 1D tensor.
TensorEngine* random_choice(TensorEngine *src, int n, bool replace, bool __GPU__) {
    if (src == NULL) {
        fprintf(stderr, "random_choice: null tensor argument\n");
        return NULL;
    }
    if (n <= 0) {
        fprintf(stderr, "random_choice: n must be positive\n");
        return NULL;
    }
    if (!replace && (size_t) n > src->size) {
        fprintf(stderr, "random_choice: cannot choose %d elements without replacement from a tensor of size %zu\n", n, src->size);
        return NULL;
    }

    // bring source data to host if needed
    f64 *src_host = NULL;
    if (src->__GPU__) {
        src_host = (f64*) malloc(sizeof(f64) * src->size);
        if (!src_host) {
            fprintf(stderr, "random_choice: memory allocation failed\n");
            return NULL;
        }
        copyDeviceToHost(src_host, src->tensor, src->size);
    } else {
        src_host = src->tensor;
    }

    f64 *result_data = (f64*) malloc(sizeof(f64) * ((size_t) n + 1));
    if (!result_data) {
        fprintf(stderr, "random_choice: memory allocation failed for result\n");
        if (src->__GPU__) free(src_host);
        return NULL;
    }

    if (replace) {
        for (int i = 0; i < n; i++) {
            size_t idx = (size_t) rand_int_below((int) src->size);
            result_data[i] = src_host[idx];
        }
    } else {
        // Partial Fisher-Yates over a temporary index pool, without
        // mutating the source tensor.
        size_t *pool = (size_t*) malloc(sizeof(size_t) * src->size);
        if (!pool) {
            fprintf(stderr, "random_choice: memory allocation failed for index pool\n");
            free(result_data);
            if (src->__GPU__) free(src_host);
            return NULL;
        }
        for (size_t i = 0; i < src->size; i++) pool[i] = i;

        size_t remaining = src->size;
        for (int i = 0; i < n; i++) {
            size_t pick = (size_t) rand_int_below((int) remaining);
            result_data[i] = src_host[pool[pick]];
            pool[pick] = pool[remaining - 1];
            remaining--;
        }
        free(pool);
    }
    result_data[n] = E;

    if (src->__GPU__) {
        free(src_host);
    }

    int shape[2];
    shape[0] = n;
    shape[1] = N;

    TensorEngine *result = build_tensor_from_host(result_data, shape, 1, __GPU__);
    free(result_data);
    return result;
}