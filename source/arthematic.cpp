// src/arthematic.cpp
#include "../include/arthematic.hpp"
#include "../__cuda__/include/kernel.cuh"
#include <cmath>

bool verify_shapes(int *s1,int *s2,size_t ndim1,size_t ndim2) {
    if (ndim1 != ndim2) return false;
    for (size_t i = 0; i < ndim1; i++) {
        if (s1[i] != s2[i]) return false;
    }
    return true;
}

static bool compute_broadcast_shape(int *s1, size_t ndim1, int *s2, size_t ndim2,
                                     int *out_shape, size_t out_ndim) {
    for (size_t i = 0; i < out_ndim; i++) {
        int d1 = (i < out_ndim - ndim1) ? 1 : s1[i - (out_ndim - ndim1)];
        int d2 = (i < out_ndim - ndim2) ? 1 : s2[i - (out_ndim - ndim2)];
        if (d1 != d2 && d1 != 1 && d2 != 1) return false;
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

// ---------------------------------------------------------------------------
// Non-broadcast ops
// ---------------------------------------------------------------------------
TensorEngine* add(TensorEngine* t1, TensorEngine* t2)
{
    if (!verify_shapes(t1->shape, t2->shape, t1->ndim, t2->ndim))
    {
        fprintf(stderr, "Tensor SHAPE are not balanced\n");
        return NULL;
    }

    if (t1->__GPU__ && t2->__GPU__)
    {
        f64 *h_cd = (double*) malloc((t1->size + 1) * sizeof(double));
        if (!h_cd)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        runDoubleOp(addKernelD, t1->tensor, t2->tensor, h_cd, t1->size);
        h_cd[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(h_cd);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(shape_terminated);
        free(h_cd);

        return result;
    }
    else
    {
        f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
        if (!result_array)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        for (size_t iter = 0; iter < t1->size; iter++)
            result_array[iter] = t1->tensor[iter] + t2->tensor[iter];

        result_array[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(result_array);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(result_array, shape_terminated, false);

        free(result_array);
        free(shape_terminated);

        return result;
    }
}



TensorEngine* sub(TensorEngine* t1, TensorEngine* t2)
{
    if (!verify_shapes(t1->shape, t2->shape, t1->ndim, t2->ndim))
    {
        fprintf(stderr, "Tensor SHAPE are not balanced\n");
        return NULL;
    }

    if (t1->__GPU__ && t2->__GPU__)
    {
        f64 *h_cd = (double*) malloc((t1->size + 1) * sizeof(double));
        if (!h_cd)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        runDoubleOp(subKernelD, t1->tensor, t2->tensor, h_cd, t1->size);
        h_cd[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(h_cd);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(shape_terminated);
        free(h_cd);

        return result;
    }
    else
    {
        f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
        if (!result_array)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        for (size_t iter = 0; iter < t1->size; iter++)
            result_array[iter] = t1->tensor[iter] - t2->tensor[iter];

        result_array[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(result_array);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(result_array, shape_terminated, false);

        free(result_array);
        free(shape_terminated);

        return result;
    }
}



TensorEngine* mlt(TensorEngine* t1, TensorEngine* t2)
{
    if (!verify_shapes(t1->shape, t2->shape, t1->ndim, t2->ndim))
    {
        fprintf(stderr, "Tensor SHAPE are not balanced\n");
        return NULL;
    }

    if (t1->__GPU__ && t2->__GPU__)
    {
        f64 *h_cd = (double*) malloc((t1->size + 1) * sizeof(double));
        if (!h_cd)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        runDoubleOp(mulKernelD, t1->tensor, t2->tensor, h_cd, t1->size);
        h_cd[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(h_cd);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(shape_terminated);
        free(h_cd);

        return result;
    }
    else
    {
        f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
        if (!result_array)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        for (size_t iter = 0; iter < t1->size; iter++)
            result_array[iter] = t1->tensor[iter] * t2->tensor[iter];

        result_array[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(result_array);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(result_array, shape_terminated, false);

        free(result_array);
        free(shape_terminated);

        return result;
    }
}


TensorEngine* divt(TensorEngine* t1, TensorEngine* t2)
{
    if (!verify_shapes(t1->shape, t2->shape, t1->ndim, t2->ndim))
    {
        fprintf(stderr, "Tensor SHAPE are not balanced\n");
        return NULL;
    }

    if (t1->__GPU__ && t2->__GPU__)
    {
        f64 *h_cd = (double*) malloc((t1->size + 1) * sizeof(double));
        if (!h_cd)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        runDoubleOp(divKernelD, t1->tensor, t2->tensor, h_cd, t1->size);
        h_cd[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(h_cd);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(shape_terminated);
        free(h_cd);

        return result;
    }
    else
    {
        f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
        if (!result_array)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }

        for (size_t iter = 0; iter < t1->size; iter++)
        {
            f64 b = t2->tensor[iter];
            result_array[iter] = (b != 0.0) ? (t1->tensor[iter] / b) : 0.0;
        }

        result_array[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        if (!shape_terminated)
        {
            free(result_array);
            return NULL;
        }

        for (size_t i = 0; i < t1->ndim; i++)
            shape_terminated[i] = t1->shape[i];

        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(result_array, shape_terminated, false);

        free(result_array);
        free(shape_terminated);

        return result;
    }
}



TensorEngine* mod(TensorEngine* t1, TensorEngine* t2)
{
    if (!verify_shapes(t1->shape, t2->shape, t1->ndim, t2->ndim))
    {
        fprintf(stderr, "Tensor SHAPE are not balanced\n");
        return NULL;
    }

    // mod has no CUDA kernel in the non-broadcast path, so CPU only
    f64 *result_array = (f64*) malloc(sizeof(f64) * (t1->size + 1));
    if (!result_array)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    for (size_t iter = 0; iter < t1->size; iter++)
    {
        f64 b = t2->tensor[iter];
        result_array[iter] = (b != 0.0) ? fmod(t1->tensor[iter], b) : 0.0;
    }

    result_array[t1->size] = E;

    int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
    if (!shape_terminated)
    {
        free(result_array);
        return NULL;
    }

    for (size_t i = 0; i < t1->ndim; i++)
        shape_terminated[i] = t1->shape[i];

    shape_terminated[t1->ndim] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, false);

    free(result_array);
    free(shape_terminated);

    return result;
}

// ---------------------------------------------------------------------------
// Broadcast ops
// ---------------------------------------------------------------------------

// Shared setup: allocates out_shape, shape_terminated, result_array.
// Returns false on any allocation or compatibility failure and frees its own allocs.
// On success, caller owns all three pointers and must free them.
static bool broadcast_setup(
    TensorEngine *t1, TensorEngine *t2,
    size_t *out_ndim_,
    int **out_shape_,
    int **shape_terminated_,
    f64 **result_array_,
    size_t *out_size_,
    size_t *pad1_, size_t *pad2_)
{
    size_t out_ndim = (t1->ndim > t2->ndim) ? t1->ndim : t2->ndim;

    int *out_shape = (int*) malloc(sizeof(int) * out_ndim);
    if (!out_shape) { fprintf(stderr, "Memory allocation failed for broadcast shape\n"); return false; }

    if (!compute_broadcast_shape(t1->shape, t1->ndim, t2->shape, t2->ndim, out_shape, out_ndim)) {
        fprintf(stderr, "Tensors are not broadcastable\n");
        free(out_shape); return false;
    }

    size_t out_size = 1;
    for (size_t i = 0; i < out_ndim; i++) out_size *= (size_t) out_shape[i];

    int *shape_terminated = (int*) malloc(sizeof(int) * (out_ndim + 1));
    if (!shape_terminated) { free(out_shape); return false; }
    for (size_t i = 0; i < out_ndim; i++) shape_terminated[i] = out_shape[i];
    shape_terminated[out_ndim] = N;

    f64 *result_array = (f64*) malloc(sizeof(f64) * (out_size + 1));
    if (!result_array) { free(out_shape); free(shape_terminated); return false; }

    *out_ndim_        = out_ndim;
    *out_shape_       = out_shape;
    *shape_terminated_= shape_terminated;
    *result_array_    = result_array;
    *out_size_        = out_size;
    *pad1_            = out_ndim - t1->ndim;
    *pad2_            = out_ndim - t2->ndim;
    return true;
}

// Shared CPU index-walk loop; op selects the scalar operation.
typedef enum { BROAD_ADD, BROAD_SUB, BROAD_MLT, BROAD_DIV, BROAD_MOD } BroadOp;

static bool broadcast_cpu_loop(
    TensorEngine *t1, TensorEngine *t2,
    f64 *result_array,
    int *out_shape, size_t out_ndim,
    size_t out_size,
    size_t pad1, size_t pad2,
    BroadOp op)
{
    size_t *strides1 = compute_strides(t1->shape, t1->ndim);
    size_t *strides2 = compute_strides(t2->shape, t2->ndim);
    size_t *idx      = (size_t*) malloc(sizeof(size_t) * out_ndim);

    if (!strides1 || !strides2 || !idx) {
        fprintf(stderr, "Memory allocation failed for broadcast bookkeeping\n");
        free(strides1); free(strides2); free(idx);
        return false;
    }

    for (size_t linear = 0; linear < out_size; linear++) {
        size_t remainder = linear;
        for (size_t d = out_ndim; d-- > 0; ) {
            idx[d] = remainder % (size_t) out_shape[d];
            remainder /= (size_t) out_shape[d];
        }
        size_t off1 = 0;
        for (size_t d = 0; d < t1->ndim; d++) {
            size_t use = (t1->shape[d] == 1) ? 0 : idx[d + pad1];
            off1 += use * strides1[d];
        }
        size_t off2 = 0;
        for (size_t d = 0; d < t2->ndim; d++) {
            size_t use = (t2->shape[d] == 1) ? 0 : idx[d + pad2];
            off2 += use * strides2[d];
        }
        f64 a = t1->tensor[off1], b = t2->tensor[off2];
        switch (op) {
            case BROAD_ADD: result_array[linear] = a + b; break;
            case BROAD_SUB: result_array[linear] = a - b; break;
            case BROAD_MLT: result_array[linear] = a * b; break;
            case BROAD_DIV: result_array[linear] = (b != 0.0) ? a / b : 0.0; break;
            case BROAD_MOD: result_array[linear] = (b != 0.0) ? fmod(a, b) : 0.0; break;
        }
    }

    free(strides1); free(strides2); free(idx);
    return true;
}

TensorEngine* add_broad(TensorEngine* t1, TensorEngine* t2) {
    size_t out_ndim, out_size, pad1, pad2;
    int *out_shape, *shape_terminated;
    f64 *result_array;

    if (!broadcast_setup(t1, t2, &out_ndim, &out_shape, &shape_terminated,
                         &result_array, &out_size, &pad1, &pad2))
        return NULL;

    if (t1->__GPU__ && t2->__GPU__) {
        runBroadcastDoubleOp(
            addBroadcastKernelD,
            t1->tensor, t2->tensor, result_array,
            t1->shape, t1->ndim, pad1,
            t2->shape, t2->ndim, pad2,
            out_shape, out_ndim, out_size);
    } else {
        if (!broadcast_cpu_loop(t1, t2, result_array, out_shape, out_ndim,
                                 out_size, pad1, pad2, BROAD_ADD)) {
            free(out_shape); free(shape_terminated); free(result_array);
            return NULL;
        }
    }

    result_array[out_size] = E;
    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__ && t2->__GPU__);
    free(result_array); free(out_shape); free(shape_terminated);
    return result;
}

TensorEngine* sub_broad(TensorEngine* t1, TensorEngine* t2) {
    size_t out_ndim, out_size, pad1, pad2;
    int *out_shape, *shape_terminated;
    f64 *result_array;

    if (!broadcast_setup(t1, t2, &out_ndim, &out_shape, &shape_terminated,
                         &result_array, &out_size, &pad1, &pad2))
        return NULL;

    if (t1->__GPU__ && t2->__GPU__) {
        runBroadcastDoubleOp(
            subBroadcastKernelD,
            t1->tensor, t2->tensor, result_array,
            t1->shape, t1->ndim, pad1,
            t2->shape, t2->ndim, pad2,
            out_shape, out_ndim, out_size);
    } else {
        if (!broadcast_cpu_loop(t1, t2, result_array, out_shape, out_ndim,
                                 out_size, pad1, pad2, BROAD_SUB)) {
            free(out_shape); free(shape_terminated); free(result_array);
            return NULL;
        }
    }

    result_array[out_size] = E;
    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__ && t2->__GPU__);
    free(result_array); free(out_shape); free(shape_terminated);
    return result;
}

TensorEngine* mlt_broad(TensorEngine* t1, TensorEngine* t2) {
    size_t out_ndim, out_size, pad1, pad2;
    int *out_shape, *shape_terminated;
    f64 *result_array;

    if (!broadcast_setup(t1, t2, &out_ndim, &out_shape, &shape_terminated,
                         &result_array, &out_size, &pad1, &pad2))
        return NULL;

    if (t1->__GPU__ && t2->__GPU__) {
        runBroadcastDoubleOp(
            mulBroadcastKernelD,
            t1->tensor, t2->tensor, result_array,
            t1->shape, t1->ndim, pad1,
            t2->shape, t2->ndim, pad2,
            out_shape, out_ndim, out_size);
    } else {
        if (!broadcast_cpu_loop(t1, t2, result_array, out_shape, out_ndim,
                                 out_size, pad1, pad2, BROAD_MLT)) {
            free(out_shape); free(shape_terminated); free(result_array);
            return NULL;
        }
    }

    result_array[out_size] = E;
    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__ && t2->__GPU__);
    free(result_array); free(out_shape); free(shape_terminated);
    return result;
}

TensorEngine* div_broad(TensorEngine* t1, TensorEngine* t2) {
    size_t out_ndim, out_size, pad1, pad2;
    int *out_shape, *shape_terminated;
    f64 *result_array;

    if (!broadcast_setup(t1, t2, &out_ndim, &out_shape, &shape_terminated,
                         &result_array, &out_size, &pad1, &pad2))
        return NULL;

    if (t1->__GPU__ && t2->__GPU__) {
        runBroadcastDoubleOp(
            divBroadcastKernelD,
            t1->tensor, t2->tensor, result_array,
            t1->shape, t1->ndim, pad1,
            t2->shape, t2->ndim, pad2,
            out_shape, out_ndim, out_size);
    } else {
        if (!broadcast_cpu_loop(t1, t2, result_array, out_shape, out_ndim,
                                 out_size, pad1, pad2, BROAD_DIV)) {
            free(out_shape); free(shape_terminated); free(result_array);
            return NULL;
        }
    }

    result_array[out_size] = E;
    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__ && t2->__GPU__);
    free(result_array); free(out_shape); free(shape_terminated);
    return result;
}

TensorEngine* mod_broad(TensorEngine* t1, TensorEngine* t2) {
    size_t out_ndim, out_size, pad1, pad2;
    int *out_shape, *shape_terminated;
    f64 *result_array;

    if (!broadcast_setup(t1, t2, &out_ndim, &out_shape, &shape_terminated,
                         &result_array, &out_size, &pad1, &pad2))
        return NULL;

    if (t1->__GPU__ && t2->__GPU__) {
        runBroadcastDoubleOp(
            modBroadcastKernelD,
            t1->tensor, t2->tensor, result_array,
            t1->shape, t1->ndim, pad1,
            t2->shape, t2->ndim, pad2,
            out_shape, out_ndim, out_size);
    } else {
        if (!broadcast_cpu_loop(t1, t2, result_array, out_shape, out_ndim,
                                 out_size, pad1, pad2, BROAD_MOD)) {
            free(out_shape); free(shape_terminated); free(result_array);
            return NULL;
        }
    }

    result_array[out_size] = E;
    TensorEngine *result = tensor(result_array, shape_terminated, t1->__GPU__ && t2->__GPU__);
    free(result_array); free(out_shape); free(shape_terminated);
    return result;
}

TensorEngine* dot_prod(TensorEngine* t1, TensorEngine* t2) {
    if (!t1 || !t2) {
        fprintf(stderr, "dot_prod: null tensor argument\n");
        return NULL;
    }

    bool use_gpu = t1->__GPU__ && t2->__GPU__;

    size_t ndim1 = t1->ndim;
    size_t ndim2 = t2->ndim;

    // ------------------------------------------------------------------
    // Determine contracted axis length (K)
    // ------------------------------------------------------------------
    int K1 = t1->shape[ndim1 - 1];
    int K2 = (ndim2 == 1) ? t2->shape[0] : t2->shape[ndim2 - 2];

    if (K1 != K2) {
        fprintf(stderr, "dot_prod: inner dimension mismatch (%d vs %d)\n", K1, K2);
        return NULL;
    }

    int K = K1;

    // ------------------------------------------------------------------
    // Compute output shape and dimensions
    // ------------------------------------------------------------------
    size_t out_ndim = 0;
    if (ndim1 == 1 && ndim2 == 1) {
        out_ndim = 1;
    } else if (ndim1 >= 2 && ndim2 == 1) {
        out_ndim = ndim1 - 1;
    } else if (ndim1 == 1 && ndim2 >= 2) {
        out_ndim = ndim2 - 1;
    } else {
        out_ndim = (ndim1 - 1) + (ndim2 - 1);
    }

    int *shape_terminated = (int*) malloc(sizeof(int) * (out_ndim + 1));
    if (!shape_terminated) {
        fprintf(stderr, "dot_prod: malloc failed for shape\n");
        return NULL;
    }

    if (ndim1 == 1 && ndim2 == 1) {
        shape_terminated[0] = 1;
    } else if (ndim1 >= 2 && ndim2 == 1) {
        for (size_t i = 0; i < ndim1 - 1; i++) {
            shape_terminated[i] = t1->shape[i];
        }
    } else if (ndim1 == 1 && ndim2 >= 2) {
        for (size_t j = 0; j < ndim2 - 2; j++) {
            shape_terminated[j] = t2->shape[j];
        }
        shape_terminated[ndim2 - 2] = t2->shape[ndim2 - 1];
    } else {
        size_t cur = 0;
        for (size_t i = 0; i < ndim1 - 1; i++) {
            shape_terminated[cur++] = t1->shape[i];
        }
        for (size_t j = 0; j < ndim2 - 2; j++) {
            shape_terminated[cur++] = t2->shape[j];
        }
        shape_terminated[cur++] = t2->shape[ndim2 - 1];
    }
    shape_terminated[out_ndim] = N;

    // ------------------------------------------------------------------
    // Compute sizes: M_total, P_total, cols, out_size
    // ------------------------------------------------------------------
    size_t M_total = 1;
    if (ndim1 >= 2) {
        for (size_t i = 0; i < ndim1 - 1; i++) {
            M_total *= (size_t)t1->shape[i];
        }
    }

    size_t P_total = 1;
    if (ndim2 >= 3) {
        for (size_t j = 0; j < ndim2 - 2; j++) {
            P_total *= (size_t)t2->shape[j];
        }
    }

    int cols = (ndim2 == 1) ? 1 : t2->shape[ndim2 - 1];
    size_t out_size = M_total * P_total * (size_t)cols;

    // ------------------------------------------------------------------
    // Allocate result buffer (+1 for E sentinel)
    // ------------------------------------------------------------------
    f64 *result_array = (f64*) malloc(sizeof(f64) * (out_size + 1));
    if (!result_array) {
        free(shape_terminated);
        fprintf(stderr, "dot_prod: malloc failed for result\n");
        return NULL;
    }

    // ------------------------------------------------------------------
    // Compute: GPU vs CPU
    // ------------------------------------------------------------------
    if (use_gpu) {
        size_t N_cols_out = P_total * (size_t)cols;
        if (P_total == 1) {
            runMatMulOp(t1->tensor, t2->tensor, result_array,
                        (int)M_total, K, (int)N_cols_out);
        } else {
            f64 *t2_reordered = (f64*) malloc(sizeof(f64) * t2->size);
            if (!t2_reordered) {
                free(result_array);
                free(shape_terminated);
                fprintf(stderr, "dot_prod: malloc failed for t2_reordered\n");
                return NULL;
            }
            for (size_t k = 0; k < (size_t)K; k++) {
                for (size_t p = 0; p < P_total; p++) {
                    for (size_t c = 0; c < (size_t)cols; c++) {
                        t2_reordered[k * N_cols_out + (p * (size_t)cols + c)] =
                            t2->tensor[p * ((size_t)K * (size_t)cols) + k * (size_t)cols + c];
                    }
                }
            }
            runMatMulOp(t1->tensor, t2_reordered, result_array,
                        (int)M_total, K, (int)N_cols_out);
            free(t2_reordered);
        }
    } else {
        for (size_t m = 0; m < M_total; m++) {
            for (size_t p = 0; p < P_total; p++) {
                for (size_t c = 0; c < (size_t)cols; c++) {
                    double acc = 0.0;
                    size_t t1_base = m * (size_t)K;
                    size_t t2_base = p * ((size_t)K * (size_t)cols) + c;
                    for (size_t k = 0; k < (size_t)K; k++) {
                        acc += t1->tensor[t1_base + k] * t2->tensor[t2_base + k * (size_t)cols];
                    }
                    result_array[(m * P_total + p) * (size_t)cols + c] = acc;
                }
            }
        }
    }

    result_array[out_size] = E;

    TensorEngine *result = tensor(result_array, shape_terminated, use_gpu);
    free(result_array);
    free(shape_terminated);
    return result;
}