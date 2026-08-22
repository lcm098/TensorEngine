#include "../include/utility.hpp"
#include <cstdio>
#include <cstdlib>


static bool validate_axes(const int *axes, size_t ndim) {
    if (!axes) return false;
    bool *seen = (bool*) calloc(ndim, sizeof(bool));
    if (!seen) return false;

    for (size_t i = 0; i < ndim; i++) {
        if (axes[i] == N) {
            free(seen);
            return false;
        }
        int a = axes[i];
        if (a < 0 || (size_t)a >= ndim || seen[a]) {
            free(seen);
            return false;
        }
        seen[a] = true;
    }
    free(seen);
    return true;
}

TensorEngine* T(TensorEngine * t, int *axes) {
    if (t == nullptr || t->tensor == nullptr) {
        fprintf(stderr, "Transpose: invalid null tensor\n");
        return NULL;
    }

    /* =========================
       1D Tensor: shape [N] -> [N, 1]
       ========================= */
    if (t->ndim == 1) {
        size_t rows = t->shape[0];
        size_t cols = 1;

        f64 *result_array = (f64*) malloc(sizeof(f64) * (t->size + 1));
        if (!result_array) {
            fprintf(stderr, "Memory allocation failed for transpose result\n");
            return NULL;
        }

        for (size_t i = 0; i < t->size; i++) {
            result_array[i] = t->tensor[i];
        }
        result_array[t->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * 3);
        if (!shape_terminated) {
            fprintf(stderr, "Memory allocation failed for new shape, while doing transpose\n");
            free(result_array);
            return NULL;
        }

        shape_terminated[0] = (int)rows;
        shape_terminated[1] = (int)cols;
        shape_terminated[2] = N;

        TensorEngine *result = tensor(result_array, shape_terminated, t->__GPU__);
        free(result_array);
        free(shape_terminated);
        return result;
    }

    /* =========================
       2D and Multidimensional Tensor (ND where ndim >= 2)
       ========================= */
    int *effective_axes = (int*) malloc(sizeof(int) * t->ndim);
    if (!effective_axes) {
        fprintf(stderr, "Memory allocation failed for effective_axes\n");
        return NULL;
    }

    if (axes == nullptr) {
        // Default: reverse all axes [ndim-1, ndim-2, ..., 0]
        for (size_t i = 0; i < t->ndim; i++) {
            effective_axes[i] = (int)(t->ndim - 1 - i);
        }
    } else {
        if (!validate_axes(axes, t->ndim)) {
            fprintf(stderr, "Transpose: invalid axes permutation provided\n");
            free(effective_axes);
            return NULL;
        }
        for (size_t i = 0; i < t->ndim; i++) {
            effective_axes[i] = axes[i];
        }
    }

    int *shape_terminated = (int*) malloc(sizeof(int) * (t->ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while doing transpose\n");
        free(effective_axes);
        return NULL;
    }

    for (size_t i = 0; i < t->ndim; i++) {
        shape_terminated[i] = t->shape[effective_axes[i]];
    }
    shape_terminated[t->ndim] = N;

    f64 *result_array = (f64*) malloc(sizeof(f64) * (t->size + 1));
    if (!result_array) {
        fprintf(stderr, "Memory allocation failed for transpose result\n");
        free(effective_axes);
        free(shape_terminated);
        return NULL;
    }

    if (t->__GPU__) {
        size_t *in_strides = (size_t*) malloc(sizeof(size_t) * t->ndim);
        if (!in_strides) {
            fprintf(stderr, "Memory allocation failed for in_strides\n");
            free(effective_axes);
            free(shape_terminated);
            free(result_array);
            return NULL;
        }
        for (size_t i = 0; i < t->ndim; i++) {
            in_strides[i] = (size_t)t->strides[i];
        }

        runTransposeNDDoubleOp(
            t->tensor,
            result_array,
            t->shape,
            in_strides,
            shape_terminated,
            effective_axes,
            t->ndim,
            t->size
        );
        free(in_strides);
    }
    else {
        // Fast path for 2D default reverse CPU
        if (t->ndim == 2 && axes == nullptr) {
            size_t rows = (size_t)t->shape[0];
            size_t cols = (size_t)t->shape[1];
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    result_array[j * rows + i] = t->tensor[i * cols + j];
                }
            }
        }
        else {
            // General ND transposition on CPU
            for (size_t linear = 0; linear < t->size; linear++) {
                size_t rem = linear;
                size_t in_offset = 0;
                for (size_t d = t->ndim; d-- > 0; ) {
                    size_t coord = rem % (size_t)shape_terminated[d];
                    rem /= (size_t)shape_terminated[d];
                    in_offset += coord * (size_t)t->strides[effective_axes[d]];
                }
                result_array[linear] = t->tensor[in_offset];
            }
        }
    }

    result_array[t->size] = E;
    TensorEngine *result = tensor(result_array, shape_terminated, t->__GPU__);
    free(effective_axes);
    free(result_array);
    free(shape_terminated);
    return result;
}

TensorEngine* transpose(TensorEngine * t, int *axes) {
    return T(t, axes);
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

    memcpy(host_data, t->tensor, sizeof(f64) * t->size);
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


TensorEngine* linspace(f64 start, f64 end, int num, bool __GPU__) {

    if (num <= 0) {
        fprintf(stderr, "linspace: num must be greater than 0\n");
        return NULL;
    }

    f64 *result_array = (f64 *)malloc(sizeof(f64) * (num + 1));

    if (result_array == NULL) {
        fprintf(stderr, "memory allocation in linspace failed\n");
        return NULL;
    }

    if (num == 1) {
        result_array[0] = start;
    } else {
        f64 step = (end - start) / (num - 1);

        for (int i = 0; i < num; i++) {
            result_array[i] = start + (i * step);
        }
    }

    result_array[num] = E;

    int shape[] = {num, N};

    TensorEngine *result = tensor(result_array, shape, __GPU__);
    free(result_array);
    return result;
}