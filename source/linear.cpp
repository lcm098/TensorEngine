#include "../include/linear.hpp"

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