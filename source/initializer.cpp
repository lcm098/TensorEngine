
#include "../include/initializer.hpp"


static void compute_fans(int shape[], size_t *fan_in, size_t *fan_out) {
    size_t ndim = calculate_ndim(shape);

    if (ndim == 1) {
        /* 1D tensor (e.g. bias vector): fan_in == fan_out == the single dim */
        *fan_in = (size_t) shape[0];
        *fan_out = (size_t) shape[0];
        return;
    }

    size_t receptive_field = 1;
    for (size_t i = 1; i + 1 < ndim; i++) {
        receptive_field *= (size_t) shape[i];
    }

    *fan_in  = (size_t) shape[0] * receptive_field;
    *fan_out = (size_t) shape[ndim - 1] * receptive_field;
}

static int rng_seeded = 0;

static void ensure_rng_seeded(void) {
    if (!rng_seeded) {
        srand((unsigned int) time(NULL));
        rng_seeded = 1;
    }
}

/* Returns a uniform double in [0.0, 1.0) */
static double rand_uniform_01(void) {
    return (double) rand() / ((double) RAND_MAX + 1.0);
}

/* Returns a uniform double in [-limit, limit] */
static double rand_uniform_range(double limit) {
    double u = rand_uniform_01(); /* [0,1) */
    return (u * 2.0 - 1.0) * limit;
}

/* Box-Muller transform: returns a value from N(0, 1) */
static double rand_standard_normal(void) {
    double u1 = rand_uniform_01();
    double u2 = rand_uniform_01();

    /* avoid log(0) */
    if (u1 < 1e-12) {
        u1 = 1e-12;
    }

    double mag = sqrt(-2.0 * log(u1));
    return mag * cos(2.0 * M_PI * u2);
}

/* Returns a value from N(0, stddev) */
static double rand_normal(double stddev) {
    return rand_standard_normal() * stddev;
}

TensorEngine* initialize_bias(int shape[], BiasInitType type, f64 param1, f64 param2, bool __GPU__) {
    ensure_rng_seeded();

    size_t ndim = calculate_ndim(shape);
    if (ndim == 0) {
        fprintf(stderr, "initialize_bias: shape must have at least 1 dimension\n");
        return NULL;
    }

    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "initialize_bias: shape dimensions must be positive\n");
            return NULL;
        }
        total *= (size_t) shape[i];
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (total + 1));
    if (!host_data) {
        fprintf(stderr, "Memory allocation failed for bias initializer data\n");
        return NULL;
    }

    switch (type) {
        case BIAS_ZEROS: {
            for (size_t i = 0; i < total; i++) host_data[i] = 0.0;
            break;
        }
        case BIAS_ONES: {
            for (size_t i = 0; i < total; i++) host_data[i] = 1.0;
            break;
        }
        case BIAS_CONSTANT: {
            for (size_t i = 0; i < total; i++) host_data[i] = param1;
            break;
        }
        case BIAS_UNIFORM: {
            double low = param1;
            double high = param2;
            if (low > high) { double tmp = low; low = high; high = tmp; }
            for (size_t i = 0; i < total; i++) {
                double u = rand_uniform_01(); /* [0,1) */
                host_data[i] = low + u * (high - low);
            }
            break;
        }
        case BIAS_NORMAL: {
            double mean = param1;
            double stddev = param2;
            for (size_t i = 0; i < total; i++) {
                host_data[i] = mean + rand_standard_normal() * stddev;
            }
            break;
        }
        case BIAS_TRUNCATED_NORMAL: {
            double mean = param1;
            double stddev = param2;
            double lower = mean - 2.0 * stddev;
            double upper = mean + 2.0 * stddev;
            for (size_t i = 0; i < total; i++) {
                double val;
                int attempts = 0;
                do {
                    val = mean + rand_standard_normal() * stddev;
                    attempts++;
                } while ((val < lower || val > upper) && attempts < 100);
                /* after 100 failed resamples, just clamp to bounds to guarantee termination */
                if (val < lower) val = lower;
                if (val > upper) val = upper;
                host_data[i] = val;
            }
            break;
        }
        default:
            fprintf(stderr, "initialize_bias: unknown BiasInitType\n");
            free(host_data);
            return NULL;
    }

    host_data[total] = E;

    int *shape_terminated = (int*) malloc(sizeof(int) * (ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while initializing bias\n");
        free(host_data);
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++) {
        shape_terminated[i] = shape[i];
    }
    shape_terminated[ndim] = N;

    TensorEngine *result = tensor(host_data, shape_terminated, __GPU__);

    free(host_data);
    free(shape_terminated);

    return result;
}


TensorEngine* initialize_tensor(int shape[], InitType type, bool __GPU__) {
    ensure_rng_seeded();

    size_t ndim = calculate_ndim(shape);
    if (ndim == 0) {
        fprintf(stderr, "initialize_tensor: shape must have at least 1 dimension\n");
        return NULL;
    }

    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "initialize_tensor: shape dimensions must be positive\n");
            return NULL;
        }
        total *= (size_t) shape[i];
    }

    size_t fan_in, fan_out;
    compute_fans(shape, &fan_in, &fan_out);

    if (fan_in == 0 || fan_out == 0) {
        fprintf(stderr, "initialize_tensor: fan_in/fan_out must be non-zero\n");
        return NULL;
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (total + 1));
    if (!host_data) {
        fprintf(stderr, "Memory allocation failed for initializer data\n");
        return NULL;
    }

    switch (type) {
        case INIT_XAVIER_UNIFORM:
        case INIT_GLOROT_UNIFORM: {
            double limit = sqrt(6.0 / (double)(fan_in + fan_out));
            for (size_t i = 0; i < total; i++) host_data[i] = rand_uniform_range(limit);
            break;
        }
        case INIT_XAVIER_NORMAL:
        case INIT_GLOROT_NORMAL: {
            double stddev = sqrt(2.0 / (double)(fan_in + fan_out));
            for (size_t i = 0; i < total; i++) host_data[i] = rand_normal(stddev);
            break;
        }
        case INIT_KAIMING_UNIFORM:
        case INIT_HE_UNIFORM: {
            double limit = sqrt(6.0 / (double) fan_in);
            for (size_t i = 0; i < total; i++) host_data[i] = rand_uniform_range(limit);
            break;
        }
        case INIT_KAIMING_NORMAL:
        case INIT_HE_NORMAL: {
            double stddev = sqrt(2.0 / (double) fan_in);
            for (size_t i = 0; i < total; i++) host_data[i] = rand_normal(stddev);
            break;
        }
        case INIT_LECUN_UNIFORM: {
            double limit = sqrt(3.0 / (double) fan_in);
            for (size_t i = 0; i < total; i++) host_data[i] = rand_uniform_range(limit);
            break;
        }
        case INIT_LECUN_NORMAL: {
            double stddev = sqrt(1.0 / (double) fan_in);
            for (size_t i = 0; i < total; i++) host_data[i] = rand_normal(stddev);
            break;
        }
        default:
            fprintf(stderr, "initialize_tensor: unknown InitType\n");
            free(host_data);
            return NULL;
    }

    host_data[total] = E;

    int *shape_terminated = (int*) malloc(sizeof(int) * (ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while initializing tensor\n");
        free(host_data);
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++) {
        shape_terminated[i] = shape[i];
    }
    shape_terminated[ndim] = N;

    TensorEngine *result = tensor(host_data, shape_terminated, __GPU__);

    free(host_data);
    free(shape_terminated);

    return result;
}