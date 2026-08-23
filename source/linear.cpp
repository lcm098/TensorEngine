#include "../include/linear.hpp"

typedef struct { 
    ActivationType type; 
} ActCtx;

static void release_act_ctx(void *ctx) { free(ctx); }

static double activation_derivative_scalar(double x, ActivationType AT) {
    switch (AT) {
        case Sigmoid:   { double s = 1.0/(1.0+exp(-x)); return s*(1.0-s); }
        case ReLU:      return x > 0.0 ? 1.0 : 0.0;
        case Tanh:      { double t = tanh(x); return 1.0 - t*t; }
        case LeakyReLU: return x > 0.0 ? 1.0 : 0.01;
        case ELU:       return x > 0.0 ? 1.0 : exp(x);
        case Swish:     { double s = 1.0/(1.0+exp(-x)); double sw = x*s; return sw + s*(1.0-sw); }
        case Linear:    return 1.0;
        default:        return 1.0; // GELU/Softmax approximated as identity here
    }
}

void activation_backward_fn(GradFn *self, TensorEngine *grad_output) {

    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;
    ActCtx *ctx = (ActCtx*) self->context;
    ActivationType AT = ctx ? ctx->type : Linear;

    f64 *deriv_host = (f64*) malloc(sizeof(f64) * (input->size + 1));
    if (!deriv_host) return;

    if (input->__GPU__) {
        f64 *host_in = (f64*) malloc(sizeof(f64) * input->size);
        copyDeviceToHost(host_in, input->tensor, input->size);
        for (size_t i = 0; i < input->size; i++) deriv_host[i] = activation_derivative_scalar(host_in[i], AT);
        free(host_in);
    } else {
        for (size_t i = 0; i < input->size; i++) deriv_host[i] = activation_derivative_scalar(input->tensor[i], AT);
    }
    deriv_host[input->size] = E;

    int *sh = (int*) malloc(sizeof(int) * (input->ndim + 1));
    for (size_t i = 0; i < input->ndim; i++) sh[i] = input->shape[i];
    sh[input->ndim] = N;

    TensorEngine *deriv = tensor(deriv_host, sh, input->__GPU__);
    free(deriv_host); free(sh);
    if (!deriv) return;

    TensorEngine *da = mlt(grad_output, deriv);
    free_tensor(deriv);
    if (!da) return;

    accumulate_grad(input, da);
    free_tensor(da);
}

Layer* Linear_Lay(
    int in_feature,
    int out_feature,
    InitType BIY,
    ActivationType AT,
    BiasInitType BIT,
    f64 bias_param1,
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

    Layer *layer = (Layer*) malloc(sizeof(Layer));
    if (layer == NULL) {
        fprintf(stderr, "Linear_Lay: memory allocation failed for layer\n");
        free_tensor(weights);
        free_tensor(bias);
        return NULL;
    }

    layer->weights = weights;
    layer->bias = bias;
    layer->activation = AT;
    layer->in_feature = in_feature;
    layer->out_feature = out_feature;

    return layer;
}

void free_layer(Layer *layer) {
    if (layer == NULL) return;

    if (layer->weights != NULL) {
        if (layer->weights->__GPU__) {
            freeDeviceMemory(layer->weights->tensor);
        } else {
            free(layer->weights->tensor);
        }
        free(layer->weights->shape);
        free(layer->weights->strides);
        free(layer->weights);
    }

    if (layer->bias != NULL) {
        if (layer->bias->__GPU__) {
            freeDeviceMemory(layer->bias->tensor);
        } else {
            free(layer->bias->tensor);
        }
        free(layer->bias->shape);
        free(layer->bias->strides);
        free(layer->bias);
    }

    free(layer);
}

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

// Applies the given activation elementwise to t, returning a NEW tensor
// built through tensor() (so shape/strides/ndim/autograd fields are all
// correctly initialized, unlike hand-rolled struct construction).

TensorEngine* apply_activation(TensorEngine *t, ActivationType AT) {

    if (t == NULL) {
        fprintf(stderr, "apply_activation: null tensor argument\n");
        return NULL;
    }

    f64 *host_data = (f64*) malloc(sizeof(f64) * (t->size + 1));
    if (host_data == NULL) {
        fprintf(stderr, "apply_activation: memory allocation failed\n");
        return NULL;
    }

    if (t->__GPU__) {
        // bring to host for elementwise activation, since no activation kernel exists yet
        copyDeviceToHost(host_data, t->tensor, t->size);
        for (size_t i = 0; i < t->size; i++) {
            host_data[i] = apply_activation_scalar(host_data[i], AT);
        }
    } else {
        for (size_t i = 0; i < t->size; i++) {
            host_data[i] = apply_activation_scalar(t->tensor[i], AT);
        }
    }
    host_data[t->size] = E;

    int *shape_terminated = (int*) malloc(sizeof(int) * (t->ndim + 1));
    if (shape_terminated == NULL) {
        fprintf(stderr, "apply_activation: memory allocation failed for shape\n");
        free(host_data);
        return NULL;
    }
    for (size_t i = 0; i < t->ndim; i++) {
        shape_terminated[i] = t->shape[i];
    }
    shape_terminated[t->ndim] = N;

    TensorEngine *result = tensor(host_data, shape_terminated, t->__GPU__);

    free(host_data);
    free(shape_terminated);

    if (result != NULL) {
        ActCtx *ctx = (ActCtx*) malloc(sizeof(ActCtx));
        if (ctx) ctx->type = AT;
        attach_unary_grad_fn(result, t, "ActivationBackward", OP_CUSTOM, activation_backward_fn, ctx, release_act_ctx);
    }

    return result;
}

