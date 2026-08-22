#ifndef TENSOR_ENGINE_UTILS_HPP
#define TENSOR_ENGINE_UTILS_HPP

#ifdef __cplusplus
extern "C" {
#endif

#include "common.hpp"
#include "gradfn.hpp"

typedef struct TensorEngine {

	f64 *tensor;
	size_t size;
	size_t ndim;
	int *shape;
	int *strides;
	bool __GPU__;
	struct TensorEngine *grad;
	bool requires_grad;
	bool is_leaf;
	GradFn *grad_fn;
	GradFn gradfn;

} TensorEngine;


TensorEngine* tensor(f64 array_[], int shape_[], bool __GPU__);
void p(TensorEngine *tensor);
void free_tensor(TensorEngine *t);

// helper methods
size_t calculate_ndim(int shape[]);
size_t calculate_elements(f64 array[]);
int* calculate_strides(int shape_[], size_t ndim);

#ifdef __cplusplus
}
#endif

#endif