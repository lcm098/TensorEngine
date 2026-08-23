#ifndef TENSOR_ENGINE_FORWARD_HPP
#define TENSOR_ENGINE_FORWARD_HPP

#include "pipeline.hpp"
#include "utils.hpp"
#include "linear.hpp"

#ifdef __cplusplus
extern "C" {
#endif

TensorEngine* forward_layer(Layer *layer, TensorEngine *input);
TensorEngine* forward(__pipeLine__ *pipe, TensorEngine *input);


#ifdef __cplusplus
}
#endif
#endif /* TENSOR_ENGINE_FORWARD_HPP */
