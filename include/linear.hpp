#ifndef TENSOR_ENGINE_LINEAR_HPP
#define TENSOR_ENGINE_LINEAR_HPP

#include "utils.hpp"
#include "./initializer.hpp"
#include "../__cuda__/include/kernel.cuh"
#include "./arthematic.hpp"

// Holds a single linear layer's configuration: weights, bias, and the
// activation to apply AFTER the forward pass (w*x + b) is computed elsewhere.
// No computation happens at construction time.
typedef struct {
    TensorEngine *weights;   // shape [in_feature, out_feature]
    TensorEngine *bias;      // shape [1, out_feature]
    ActivationType activation;
    int in_feature;
    int out_feature;
} Layer;

#ifdef __cplusplus
extern "C" {
#endif

// Builds and returns a Layer config: initializes weights + bias tensors,
// but does NOT combine them or apply the activation. That happens later
// in a separate forward-pass function.

Layer* Linear_Lay(
    int in_feature,
    int out_feature,
    InitType BIY,          /* weight initializer */
    ActivationType AT,
    BiasInitType BIT,      /* bias initializer */
    f64 bias_param1,       /* meaning depends on BIT, see initialize_bias */
    f64 bias_param2,
    bool __GPU__
);

void free_layer(Layer *layer);

#ifdef __cplusplus
}
#endif
#endif /* TENSOR_ENGINE_LINEAR_HPP */