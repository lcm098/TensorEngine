
#include "../include/arthematic.hpp"
#include "../include/common.hpp"
#include "../__cuda__/include/kernel.cuh"


bool verify_shapes(int *s1,int *s2,size_t ndim1,size_t ndim2) {
    if (ndim1 != ndim2)
        return false;

    for (size_t i = 0; i < ndim1; i++) {
        if (s1[i] != s2[i])
            return false;
    }

    return true;
}

TensorEngine* add(TensorEngine* t1, TensorEngine* t2) {
    if (!verify_shapes(t1->shape, t2->shape, t1->ndim,t2->ndim))  {
        fprintf(stderr, "Tensor SHAPE are not balanced\n");
        return NULL;
    }

    if (t1->__GPU__ && t2->__GPU__) {
        // +1 slot for the E (-INFINITY) sentinel calculate_elements() scans for
        
        f64 *h_cd = (double *)malloc((t1->size + 1) * sizeof(double));

        if (!h_cd) {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        runDoubleOp(addKernelD, t1->tensor, t2->tensor, h_cd, t1->size);
        h_cd[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        
        if(!shape_terminated) {
            fprintf(stderr, "Memory allocation failed for new shape, while doing tensor add");
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++) {
            shape_terminated[i] = t1->shape[i];
        }
        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(shape_terminated);
        free(h_cd);

        return result;
    }
    else {
        
        t1->__GPU__ = false;
        t2->__GPU__ = false;

        f64 *result_array = (f64*)malloc(sizeof(f64) * (t1->size+1));

        if(!result_array) {
            fprintf(stderr, "New Memory allocation failed, for Tensor add operation");
            return NULL;
        }

        for(size_t iter=0; iter < t1->size; iter++) {
            result_array[iter] = t1->tensor[iter] + t2->tensor[iter];
        }
        result_array[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if(!shape_terminated) {
            fprintf(stderr, "Memory allocation failed for new shape, while doing tensor add");
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++) {
            shape_terminated[i] = t1->shape[i];
        }
        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(result_array, shape_terminated, false);
        free(result_array);
        free(shape_terminated);

        return result;
    }
}

static bool compute_broadcast_shape(int *s1, size_t ndim1, int *s2, size_t ndim2,
                                     int *out_shape, size_t out_ndim) {
    for (size_t i = 0; i < out_ndim; i++) {
        int d1 = (i < out_ndim - ndim1) ? 1 : s1[i - (out_ndim - ndim1)];
        int d2 = (i < out_ndim - ndim2) ? 1 : s2[i - (out_ndim - ndim2)];

        if (d1 != d2 && d1 != 1 && d2 != 1)
            return false;

        out_shape[i] = (d1 > d2) ? d1 : d2;
    }
    return true;
}

static size_t* compute_strides(int *shape, size_t ndim) {
    size_t *strides = (size_t*) malloc(sizeof(size_t) * ndim);
    if (!strides) return NULL;
    size_t acc = 1;
    for (size_t i = ndim; i-- > 0; ) {
        strides[i] = acc;
        acc *= (size_t) shape[i];
    }
    return strides;
}

TensorEngine* add_broad(TensorEngine* t1, TensorEngine* t2) {
    size_t out_ndim = (t1->ndim > t2->ndim) ? t1->ndim : t2->ndim;

    int *out_shape = (int*) malloc(sizeof(int) * out_ndim);
    if (!out_shape) {
        fprintf(stderr, "Memory allocation failed for broadcast shape\n");
        return NULL;
    }

    if (!compute_broadcast_shape(t1->shape, t1->ndim, t2->shape, t2->ndim, out_shape, out_ndim)) {
        fprintf(stderr, "Tensors are not broadcastable\n");
        free(out_shape);
        return NULL;
    }

    size_t out_size = 1;
    for (size_t i = 0; i < out_ndim; i++) {
        out_size *= (size_t) out_shape[i];
    }

    size_t pad1 = out_ndim - t1->ndim;
    size_t pad2 = out_ndim - t2->ndim;

    int *shape_terminated = (int*) malloc(sizeof(int) * (out_ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "Memory allocation failed for new shape, while doing broadcast tensor add\n");
        free(out_shape);
        return NULL;
    }
    for (size_t i = 0; i < out_ndim; i++) shape_terminated[i] = out_shape[i];
    shape_terminated[out_ndim] = N;

    if (t1->__GPU__ && t2->__GPU__) {
        // +1 slot for the E sentinel, same convention as add()
        f64 *h_cd = (f64*) malloc((out_size + 1) * sizeof(double));
        if (!h_cd) {
            fprintf(stderr, "Memory allocation failed\n");
            free(out_shape);
            free(shape_terminated);
            return NULL;
        }

        runBroadcastDoubleOp(
            t1->tensor, t2->tensor, h_cd,
            t1->shape, t1->ndim, pad1,
            t2->shape, t2->ndim, pad2,
            out_shape, out_ndim,
            out_size);

        h_cd[out_size] = E;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(h_cd);
        free(out_shape);
        free(shape_terminated);

        return result;
    }
    else {
        t1->__GPU__ = false;
        t2->__GPU__ = false;

        f64 *result_array = (f64*) malloc(sizeof(f64) * (out_size + 1));
        if (!result_array) {
            fprintf(stderr, "Memory allocation failed for broadcast add result\n");
            free(out_shape);
            free(shape_terminated);
            return NULL;
        }

        size_t *strides1 = compute_strides(t1->shape, t1->ndim);
        size_t *strides2 = compute_strides(t2->shape, t2->ndim);
        size_t *idx = (size_t*) malloc(sizeof(size_t) * out_ndim);

        if (!strides1 || !strides2 || !idx) {
            fprintf(stderr, "Memory allocation failed for broadcast bookkeeping\n");
            free(out_shape); free(shape_terminated); free(result_array);
            free(strides1); free(strides2); free(idx);
            return NULL;
        }

        for (size_t linear = 0; linear < out_size; linear++) {
            size_t remainder = linear;
            for (size_t d = out_ndim; d-- > 0; ) {
                idx[d] = remainder % (size_t) out_shape[d];
                remainder /= (size_t) out_shape[d];
            }

            size_t off1 = 0;
            for (size_t d = 0; d < t1->ndim; d++) {
                size_t use_index = (t1->shape[d] == 1) ? 0 : idx[d + pad1];
                off1 += use_index * strides1[d];
            }

            size_t off2 = 0;
            for (size_t d = 0; d < t2->ndim; d++) {
                size_t use_index = (t2->shape[d] == 1) ? 0 : idx[d + pad2];
                off2 += use_index * strides2[d];
            }

            result_array[linear] = t1->tensor[off1] + t2->tensor[off2];
        }
        result_array[out_size] = E;

        TensorEngine *result = tensor(result_array, shape_terminated, false);

        free(result_array);
        free(strides1);
        free(strides2);
        free(idx);
        free(out_shape);
        free(shape_terminated);

        return result;
    }
}