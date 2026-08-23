
#include "../include/linear.hpp"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static f64 apply_activation_scalar(f64 x, ActivationType AT) {
    switch (AT) {
        case Sigmoid:   return 1.0 / (1.0 + exp(-x));
        case ReLU:      return x > 0.0 ? x : 0.0;
        case Tanh:      return tanh(x);
        case Softmax:   return exp(x); /* caller must normalize across the row afterward */
        case LeakyReLU: return x > 0.0 ? x : 0.01 * x;
        case ELU:       return x > 0.0 ? x : (exp(x) - 1.0);
        case GELU:      return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        case Swish:     return x / (1.0 + exp(-x));
        case Linear:    return x;
        default:        return x;
    }
}

TensorEngine* Linear_Lay(
    int in_feature,
    int out_feature,
    InitType BIY,          /* weight initializer */
    ActivationType AT,
    BiasInitType BIT,      /* bias initializer */
    f64 bias_param1,       /* meaning depends on BIT, see initialize_bias */
    f64 bias_param2,
    bool __GPU__
) {
    if (in_feature <= 0 || out_feature <= 0) {
        fprintf(stderr, "Linear_Lay: in_feature/out_feature must be positive\n");
        return NULL;
    }

    int weight_shape[] = {in_feature, out_feature, N};
    TensorEngine *weights = initialize_tensor(weight_shape, BIY, __GPU__);
    if (weights == NULL) {
        fprintf(stderr, "Linear_Lay: failed to initialize weights\n");
        return NULL;
    }

    int bias_shape[] = {1, out_feature, N};
    TensorEngine *bias = initialize_bias(bias_shape, BIT, bias_param1, bias_param2, __GPU__);
    if (bias == NULL) {
        fprintf(stderr, "Linear_Lay: failed to initialize bias\n");
        free_tensor(weights);
        return NULL;
    }

    /* broadcast-add bias across every row of weights: [in,out] + [1,out] -> [in,out] */
    TensorEngine *combined = add_broad(weights, bias);
    free_tensor(weights);
    free_tensor(bias);

    if (combined == NULL) {
        fprintf(stderr, "Linear_Lay: failed to combine weights and bias\n");
        return NULL;
    }

    /* apply activation elementwise, in place on host memory */
    bool was_gpu = combined->__GPU__;

    if (was_gpu) {

        /* bring to host for elementwise activation, since no activation kernel exists yet */
        f64 *host_copy = (f64*) malloc(sizeof(f64) * combined->size);
        if (!host_copy) {
            fprintf(stderr, "Linear_Lay: memory allocation failed for activation buffer\n");
            free_tensor(combined);
            return NULL;
        }

        cudaMemcpy(host_copy, combined->tensor, sizeof(f64) * combined->size, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < combined->size; i++) {
            host_copy[i] = apply_activation_scalar(host_copy[i], AT);
        }

        cudaMemcpy(combined->tensor, host_copy, sizeof(f64) * combined->size, cudaMemcpyHostToDevice);
        free(host_copy);
    } else {

        for (size_t i = 0; i < combined->size; i++) {
            combined->tensor[i] = apply_activation_scalar(combined->tensor[i], AT);
        }
    }

    return combined;
}