
#ifndef TENSOR_ENGINE_UTILITY_HPP
#define TENSOR_ENGINE_UTILITY_HPP

#include "common.hpp"
#include "utils.hpp"
#include "../__cuda__/include/kernel.cuh"

TensorEngine* T(TensorEngine * t);
TensorEngine* arange(f64 start, f64 end, f64 step, bool gpu);
TensorEngine* reshape(TensorEngine* t, int *new_shape, size_t new_ndim);


#endif