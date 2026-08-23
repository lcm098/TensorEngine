
#include "../include/forward.hpp"
#include <cstdlib>

TensorEngine* forward_layer(Layer *layer, TensorEngine *input) {

    if (layer == NULL || input == NULL) {
        fprintf(stderr, "forward_layer: null argument\n");
        return NULL;
    }

    if (layer->weights == NULL || layer->bias == NULL) {
        fprintf(stderr, "forward_layer: layer has no weights/bias\n");
        return NULL;
    }

    // z = input @ weights   (dot_prod handles [batch,in]x[in,out] -> [batch,out],
    //                        or [in] x [in,out] -> [out] for a single sample)

    TensorEngine *z = dot_prod(input, layer->weights);
    if (z == NULL) {
        fprintf(stderr, "forward_layer: dot_prod failed\n");
        return NULL;
    }

    // z = z + bias   (broadcast add: [batch,out] + [1,out] -> [batch,out])
    TensorEngine *z_biased = add_broad(z, layer->bias);
    if (z_biased == NULL) {
        fprintf(stderr, "forward_layer: add_broad failed\n");
        return NULL;
    }

    // apply activation elementwise
    TensorEngine *output = apply_activation(z_biased, layer->activation);
    if (output == NULL) {
        fprintf(stderr, "forward_layer: activation failed\n");
        return NULL;
    }

    return output;
}


TensorEngine* forward(__pipeLine__ *pipe, TensorEngine *input) {

    if (pipe == NULL) {
        fprintf(stderr, "forward: null pipeline argument\n");
        return NULL;
    }
    if (input == NULL) {
        fprintf(stderr, "forward: null input argument\n");
        return NULL;
    }
    if (pipe->depth == 0) {
        fprintf(stderr, "forward: pipeline has no layers\n");
        return NULL;
    }

    TensorEngine *current = input;

    for (size_t i = 0; i < pipe->depth; i++) {
        Layer *layer = pipe->layers[i];

        if (layer == NULL) {
            fprintf(stderr, "forward: null layer at index %zu\n", i);
            return NULL;
        }

        TensorEngine *next = forward_layer(layer, current);
        if (next == NULL) {
            fprintf(stderr, "forward: layer %zu failed\n", i);
            return NULL;
        }

        current = next;
    }

    // final layer's output
    return current;
}