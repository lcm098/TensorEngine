
#include "../include/common.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include "../include/utils.hpp"


int* calculate_strides(int shape_[], size_t ndim)
{
	int *strides = (int*) malloc(sizeof(int) * ndim);

	if (strides == nullptr) {
		fprintf(stderr, "memory allocation failed for strides");
		return NULL;
	}

	strides[ndim - 1] = 1;

	for (int iter = ndim - 2; iter >= 0; iter--) {
		strides[iter] =
			strides[iter + 1] *
			shape_[iter + 1];
	}

	return strides;
}

size_t calculate_ndim(int shape[])
{
	size_t ndim = 0;

	while (shape[ndim] != -1) {
		ndim++;
	}

	return ndim;
}

size_t calculate_elements(f64 array[])
{
	size_t elements = 0;
	while (array[elements] != -INFINITY) {
		elements++;
	}
	return elements;
}

void free_tensor(TensorEngine *t) {
    if (t == nullptr) return;
    free(t->tensor);
    free(t->shape);
    free(t->strides);
    delete t;
}

TensorEngine* tensor(f64 array_[], int shape_[], bool __GPU__)
{
	size_t ndim = calculate_ndim(shape_);

	size_t FLAG = 1;

	for (size_t iter = 0; iter < ndim; iter++) {
		FLAG *= shape_[iter];
	}
	
	size_t elements = calculate_elements(array_);

	if (FLAG != elements) {
		fprintf(stderr, "size is unbalance according to given tensor\n");
		return NULL;
	}

	TensorEngine *engine = new TensorEngine;

	engine->size = elements;
	engine->ndim = ndim;

	engine->tensor = (f64*) malloc(sizeof(f64) * elements);

	if (engine->tensor == nullptr) {
		fprintf(stderr, "memory allocation failed for new tensor\n");
		delete engine;
		return NULL;
	}

	for (size_t iter = 0; iter < elements; iter++) {
		engine->tensor[iter] = array_[iter];
	}

	engine->shape = (int*) malloc(sizeof(int) * ndim);

	if (engine->shape == nullptr) {
		fprintf(stderr, "memory allocation failed for tensor shape\n");
		free(engine->tensor);
		return NULL;
	}

	for (size_t iter = 0; iter < ndim; iter++) {
		engine->shape[iter] = shape_[iter];
	}

	engine->strides = calculate_strides(
		engine->shape,
		engine->ndim
	);

	if (engine->strides == nullptr) {
		fprintf(stderr, "memory allocation failed for tensor strides\n");
		free(engine->shape);
		free(engine->tensor);
		return NULL;
	}

	if (__GPU__) {
		engine->__GPU__ = true;
	} else {
		engine->__GPU__ = false;
	}

	return engine;
}

void print_recursive(f64 *data,int *shape,int *strides,size_t ndim,size_t dim,size_t offset) {
	if (dim == ndim - 1) {

		printf("[");

		for (size_t iter = 0; iter < (size_t)shape[dim]; iter++) {

			printf("%lf", data[offset + iter * strides[dim]]);

			if (iter + 1 < (size_t)shape[dim]) {
				printf(", ");
			}
		}

		printf("]");
		return;
	}

	printf("[");

	for (size_t iter = 0; iter < (size_t)shape[dim]; iter++) {

		printf("\n");

		for (size_t space = 0; space <= dim; space++) {
			printf("  ");
		}

		print_recursive(data,shape,strides,ndim,dim + 1,offset + iter * strides[dim]);

		if (iter + 1 < (size_t)shape[dim]) {
			printf(",");
		}
	}

	printf("\n");

	for (size_t space = 0; space < dim; space++) {
		printf("  ");
	}

	printf("]");
}

void p(TensorEngine* tensor) {

	printf("======================================\n");

	if (tensor == NULL || tensor->tensor == NULL) {
		fprintf(stderr, "invalid tensor\n");
		return;
	}

	print_recursive(tensor->tensor,tensor->shape,tensor->strides,tensor->ndim,0,0);
	printf("\n");
	printf("size = %zu\n", tensor->size);
	printf("ndim = %zu\n", tensor->ndim);
	printf("shape = [");

	for (size_t iter = 0; iter < tensor->ndim; iter++) {
		printf("%d", tensor->shape[iter]);

		if (iter + 1 < tensor->ndim) {
			printf(", ");
		}
	}

	printf("]\n");
	printf("strides = [");
	for (size_t iter = 0; iter < tensor->ndim; iter++) {
		printf("%d", tensor->strides[iter]);

		if (iter + 1 < tensor->ndim) {
			printf(", ");
		}
	}
	printf("]\n");
	const char *status = tensor->__GPU__ ? "true":"false";
	printf("__gpu__ [%s]\n", status);
	printf("======================================\n");

}
