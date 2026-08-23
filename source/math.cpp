
#include "../include/math.hpp"
#include "../include/gradfn.hpp"
#include "../include/utils.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define PI 3.141592653589793238462643383279502884
#define TWO_PI (2.0 * PI)
#define HALF_PI (PI / 2.0)
#define QUARTER_PI (PI / 4.0)
#define DEFAULT_TERMS 10

[[maybe_unused]] static double degrees_to_radians(double degrees)
{
    return degrees * (PI / 180.0);
}

// Taylor expansion for sin(x) on small interval [-pi/4, pi/4]
static double taylor_sin(double x, int TERMS)
{
    double term = x;
    double sum = x;

    for (int n = 0; n < TERMS; n++)
    {
        term = term * (-x * x)
             / ((2.0 * n + 2.0) * (2.0 * n + 3.0));

        sum += term;
    }

    return sum;
}

// Taylor expansion for cos(x) on small interval [-pi/4, pi/4]
static double taylor_cos(double x, int TERMS)
{
    double term = 1.0;
    double sum = 1.0;

    for (int n = 0; n < TERMS; n++)
    {
        term = term * (-x * x)
             / ((2.0 * n + 1.0) * (2.0 * n + 2.0));

        sum += term;
    }

    return sum;
}

// Full range-reduced sin(x)
static double calculate_sin(double x, int TERMS)
{
    if (std::isnan(x) || std::isinf(x)) return std::sin(x);

    // 1. Modulo 2*pi into [-pi, pi]
    double r = std::fmod(x, TWO_PI);
    if (r > PI) {
        r -= TWO_PI;
    } else if (r < -PI) {
        r += TWO_PI;
    }

    // 2. Reduce to [-pi/2, pi/2] using symmetry
    if (r > HALF_PI) {
        r = PI - r;
    } else if (r < -HALF_PI) {
        r = -PI - r;
    }

    // 3. Reduce to [-pi/4, pi/4] using quadrant identities
    if (r > QUARTER_PI) {
        return taylor_cos(HALF_PI - r, TERMS);
    } else if (r < -QUARTER_PI) {
        return -taylor_cos(HALF_PI + r, TERMS);
    } else {
        return taylor_sin(r, TERMS);
    }
}

// Full range-reduced cos(x)
static double calculate_cos(double x, int TERMS)
{
    if (std::isnan(x) || std::isinf(x)) return std::cos(x);

    // 1. Modulo 2*pi into [-pi, pi], then symmetry cos(-x) = cos(x)
    double r = std::fmod(x, TWO_PI);
    if (r > PI) {
        r -= TWO_PI;
    } else if (r < -PI) {
        r += TWO_PI;
    }
    r = std::fabs(r); // r in [0, pi]

    double sign = 1.0;
    // 2. Reduce to [0, pi/2] using cos(pi - r) = -cos(r)
    if (r > HALF_PI) {
        r = PI - r;
        sign = -1.0;
    }

    // 3. Reduce to [0, pi/4] using cos(r) = sin(pi/2 - r)
    if (r > QUARTER_PI) {
        return sign * taylor_sin(HALF_PI - r, TERMS);
    } else {
        return sign * taylor_cos(r, TERMS);
    }
}

TensorEngine* _cos(TensorEngine *t1) {
    if (t1 == nullptr || t1->tensor == nullptr) {
        fprintf(stderr, "_cos: null tensor argument\n");
        return NULL;
    }

    f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
    if (result_array == nullptr) {
        fprintf(stderr, "Memory allocation failed for tensor data, in _cos function\n");
        return NULL;
    }

    for (size_t idx = 0; idx < t1->size; idx++) {
        result_array[idx] = calculate_cos(t1->tensor[idx], DEFAULT_TERMS);
    }
    result_array[t1->size] = E;

    int shape_len = (int) t1->ndim;
    int *shape_terminated = (int*) malloc(sizeof(int) * (shape_len + 1));
    if (shape_terminated == nullptr) {
        fprintf(stderr, "Memory allocation failed for tensor shape, in _cos function\n");
        free(result_array);
        return NULL;
    }

    for (int i = 0; i < shape_len; i++) {
        shape_terminated[i] = t1->shape[i];
    }
    shape_terminated[shape_len] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__);
    free(result_array);
    free(shape_terminated);

    if (result != nullptr) {
        attach_unary_grad_fn(result, t1, "CosBackward", OP_COS, _cos_backward_fn, nullptr, nullptr);
    }

    return result;
}

TensorEngine* _sin(TensorEngine *t1) {
    if (t1 == nullptr || t1->tensor == nullptr) {
        fprintf(stderr, "_sin: null tensor argument\n");
        return NULL;
    }

    f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
    if (result_array == nullptr) {
        fprintf(stderr, "Memory allocation failed for tensor data, in _sin function\n");
        return NULL;
    }

    for (size_t idx = 0; idx < t1->size; idx++) {
        result_array[idx] = calculate_sin(t1->tensor[idx], DEFAULT_TERMS);
    }
    result_array[t1->size] = E;

    int shape_len = (int) t1->ndim;
    int *shape_terminated = (int*) malloc(sizeof(int) * (shape_len + 1));
    if (shape_terminated == nullptr) {
        fprintf(stderr, "Memory allocation failed for tensor shape, in _sin function\n");
        free(result_array);
        return NULL;
    }

    for (int i = 0; i < shape_len; i++) {
        shape_terminated[i] = t1->shape[i];
    }
    shape_terminated[shape_len] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__);
    free(result_array);
    free(shape_terminated);

    if (result != nullptr) {
        attach_unary_grad_fn(result, t1, "SinBackward", OP_SIN, _sin_backward_fn, nullptr, nullptr);
    }

    return result;
}

TensorEngine* _clamp(TensorEngine *t1, f64 High, f64 Low) {
    if (t1 == nullptr || t1->tensor == nullptr) {
        fprintf(stderr, "_clamp: null tensor argument\n");
        return NULL;
    }

    f64 min_val = (Low < High) ? Low : High;
    f64 max_val = (Low < High) ? High : Low;

    f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
    if (result_array == nullptr) {
        fprintf(stderr, "Memory allocation failed for tensor data, in _clamp function\n");
        return NULL;
    }

    for (size_t idx = 0; idx < t1->size; idx++) {
        f64 val = t1->tensor[idx];
        if (val < min_val) {
            result_array[idx] = min_val;
        } else if (val > max_val) {
            result_array[idx] = max_val;
        } else {
            result_array[idx] = val;
        }
    }
    result_array[t1->size] = E;

    int shape_len = (int) t1->ndim;
    int *shape_terminated = (int*) malloc(sizeof(int) * (shape_len + 1));
    if (shape_terminated == nullptr) {
        fprintf(stderr, "Memory allocation failed for tensor shape, in _clamp function\n");
        free(result_array);
        return NULL;
    }

    for (int i = 0; i < shape_len; i++) {
        shape_terminated[i] = t1->shape[i];
    }
    shape_terminated[shape_len] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__);
    free(result_array);
    free(shape_terminated);

    if (result != nullptr) {
        attach_clamp_grad_fn(result, t1, min_val, max_val);
    }

    return result;
}