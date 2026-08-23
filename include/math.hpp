#ifndef TENSOR_ENGINE_MATH_HPP
#define TENSOR_ENGINE_MATH_HPP

#include "utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

TensorEngine* _cos(TensorEngine *t1);
TensorEngine* _sin(TensorEngine *t1);
TensorEngine* _clamp(TensorEngine *t1, f64 High, f64 Low);

#ifdef __cplusplus
}
#endif
#endif /* TENSOR_ENGINE_MATH_HPP */
