#ifndef TENSOR_ENGINE_LINEAR_HPP
#define TENSOR_ENGINE_LINEAR_HPP

#include "utils.hpp"
#include "./initializer.hpp"
#include "../__cuda__/include/kernel.cuh"
#include "./arthematic.hpp"

#ifdef __cplusplus
extern "C" {
#endif

TensorEngine* Linear_Lay(
    int in_feature,
    int out_feature,
    InitType BIY,          /* weight initializer */
    ActivationType AT,
    BiasInitType BIT,      /* bias initializer */
    f64 bias_param1,       /* meaning depends on BIT, see initialize_bias */
    f64 bias_param2,
    bool __GPU__
);


#ifdef __cplusplus
}
#endif
#endif /* TENSOR_ENGINE_LINEAR_HPP */
