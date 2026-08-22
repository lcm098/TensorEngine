
#ifndef TENSOR_ENGINE_UTILITY_HPP
#define TENSOR_ENGINE_UTILITY_HPP

#include "common.hpp"
#include "utils.hpp"
#include "../__cuda__/include/kernel.cuh"

TensorEngine* T(TensorEngine * t, int *axes = nullptr);
TensorEngine* transpose(TensorEngine * t, int *axes = nullptr);
TensorEngine* arange(f64 start, f64 end, f64 step, bool gpu);
TensorEngine* reshape(TensorEngine* t, int *new_shape, size_t new_ndim);
TensorEngine* linspace(f64 start, f64 end, int num, bool __GPU__) ;
TensorEngine* slice(TensorEngine* t, int indices[][3], size_t num_slices = 0, int offset = 0);

#endif