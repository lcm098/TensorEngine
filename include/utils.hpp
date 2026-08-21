#ifndef TENSOR_ENGINE_UTILS_HPP
#define TENSOR_ENGINE_UTILS_HPP

#ifdef __cplusplus
extern "C" {
#endif

#include "common.hpp"

typedef struct GradFn {
} GradFn;


typedef struct TensorEngine {

	f64 *tensor;
	size_t size;
	size_t ndim;
	int *shape;
	int *strides;
	bool __GPU__;
	GradFn gradfn;

} TensorEngine;


TensorEngine* tensor(f64 array_[], int shape_[], bool __GPU__);
void p(TensorEngine *tensor);
void free_tensor(TensorEngine *t);

#ifdef __cplusplus
}
#endif

#endif