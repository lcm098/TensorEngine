
#include "../include/utility.hpp"


TensorEngine* T(TensorEngine * t) {
    if (t->ndim != 2) {
        fprintf(stderr, "Transpose only supports 2D tensors\n");
        return NULL;
    }

    size_t rows = (size_t) t->shape[0];
    size_t cols = (size_t) t->shape[1];

    // +1 slot for the E (-INFINITY) sentinel
    f64 *result_array = (f64*) malloc(sizeof(f64) * (t->size + 1));
    if (!result_array) {
        fprintf(stderr, "Memory allocation failed for transpose result\n");
        return NULL;
    }

    if (t->__GPU__) {
        runTransposeDoubleOp(t->tensor, result_array, rows, cols);
    }
    else {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result_array[j * rows + i] = t->tensor[i * cols + j];
            }
        }
    }

    result_array[t->size] = E;

    // transposed shape: [cols, rows], N-terminated
    int *shape_terminated = (int*) malloc(sizeof(int) * (t->ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while doing transpose\n");
        free(result_array);
        return NULL;
    }

    shape_terminated[0] = (int) cols;
    shape_terminated[1] = (int) rows;
    shape_terminated[2] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, t->__GPU__);

    free(result_array);
    free(shape_terminated);

    return result;
}

TensorEngine* arange(f64 start, f64 end, f64 step, bool gpu) {
    if (step == 0) {
        fprintf(stderr, "arange: step must not be zero\n");
        return NULL;
    }

    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        fprintf(stderr, "arange: step direction does not match start/end range\n");
        return NULL;
    }

    // number of elements, ceil((end - start) / step)
    size_t size = (size_t) ceil((end - start) / step);

    // +1 slot for the E (-INFINITY) sentinel
    f64 *result_array = (f64*) malloc(sizeof(f64) * (size + 1));
    if (!result_array) {
        fprintf(stderr, "Memory allocation failed for arange result\n");
        return NULL;
    }

    for (size_t i = 0; i < size; i++) {
        result_array[i] = start + (f64) i * step;
    }
    result_array[size] = E;

    int *shape_terminated = (int*) malloc(sizeof(int) * 2);
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while doing arange\n");
        free(result_array);
        return NULL;
    }

    shape_terminated[0] = (int) size;
    shape_terminated[1] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, gpu);

    free(result_array);
    free(shape_terminated);

    return result;
}

TensorEngine* reshape(TensorEngine* t, int *new_shape, size_t new_ndim) {
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        if (new_shape[i] <= 0) {
            fprintf(stderr, "reshape: shape dimensions must be positive\n");
            return NULL;
        }
        new_size *= (size_t) new_shape[i];
    }

    if (new_size != t->size) {
        fprintf(stderr, "reshape: cannot reshape tensor of size %zu into shape with size %zu\n",
                t->size, new_size);
        return NULL;
    }

    // +1 slot for the E (-INFINITY) sentinel
    f64 *host_data = (f64*) malloc(sizeof(f64) * (t->size + 1));
    if (!host_data) {
        fprintf(stderr, "Memory allocation failed for reshape data\n");
        return NULL;
    }

    if (t->__GPU__) {
        // t->tensor is a device pointer; bring it back to host before rewrapping
        cudaMemcpy(host_data, t->tensor, sizeof(f64) * t->size, cudaMemcpyDeviceToHost);
    }
    else {
        memcpy(host_data, t->tensor, sizeof(f64) * t->size);
    }
    host_data[t->size] = E;

    int *shape_terminated = (int*) malloc(sizeof(int) * (new_ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while doing reshape\n");
        free(host_data);
        return NULL;
    }

    for (size_t i = 0; i < new_ndim; i++) {
        shape_terminated[i] = new_shape[i];
    }
    shape_terminated[new_ndim] = N;

    TensorEngine *result = tensor(host_data, shape_terminated, t->__GPU__);

    free(host_data);
    free(shape_terminated);

    return result;
}