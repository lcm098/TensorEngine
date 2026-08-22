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

    // ------------------------------------------------------------------
    // Determine M, inner (K), cols and output shape from ndim combination
    // ------------------------------------------------------------------
    int rows_out = 0, inner = 0, cols_out = 0;
    int out_ndim = 0;

    if (t1->ndim == 1 && t2->ndim == 1) {
        // vector · vector → scalar returned as shape [1]
        if (t1->size != t2->size) {
            fprintf(stderr, "dot_prod: 1D size mismatch (%zu vs %zu)\n",
                    t1->size, t2->size);
            return NULL;
        }
        rows_out = 1;
        inner    = (int)t1->size;
        cols_out = 1;
        out_ndim = 1;   // result shape [1]
    }
    else if (t1->ndim == 2 && t2->ndim == 2) {
        // matrix × matrix
        if (t1->shape[1] != t2->shape[0]) {
            fprintf(stderr, "dot_prod: shape mismatch [%d,%d] × [%d,%d]\n",
                    t1->shape[0], t1->shape[1], t2->shape[0], t2->shape[1]);
            return NULL;
        }
        rows_out = t1->shape[0];
        inner    = t1->shape[1];
        cols_out = t2->shape[1];
        out_ndim = 2;
    }
    else if (t1->ndim == 1 && t2->ndim == 2) {
        // vector × matrix  [K] × [K, cols] → [cols]
        if ((int)t1->size != t2->shape[0]) {
            fprintf(stderr, "dot_prod: 1D×2D shape mismatch (%zu vs %d)\n",
                    t1->size, t2->shape[0]);
            return NULL;
        }
        rows_out = 1;
        inner    = (int)t1->size;
        cols_out = t2->shape[1];
        out_ndim = 1;   // result shape [cols]
    }
    else if (t1->ndim == 2 && t2->ndim == 1) {
        // matrix × vector  [M, K] × [K] → [M]
        if (t1->shape[1] != (int)t2->size) {
            fprintf(stderr, "dot_prod: 2D×1D shape mismatch (%d vs %zu)\n",
                    t1->shape[1], t2->size);
            return NULL;
        }
        rows_out = t1->shape[0];
        inner    = t1->shape[1];
        cols_out = 1;
        out_ndim = 1;   // result shape [M]
    }
    else {
        fprintf(stderr, "dot_prod: unsupported ndim combination (%zu, %zu); "
                        "only 1D/2D tensors are supported\n",
                t1->ndim, t2->ndim);
        return NULL;
    }

    size_t out_size = (size_t)rows_out * cols_out;

    // ------------------------------------------------------------------
    // Allocate result buffer (+1 for E sentinel)
    // ------------------------------------------------------------------
    f64 *result_array = (f64*) malloc(sizeof(f64) * (out_size + 1));
    if (!result_array) {
        fprintf(stderr, "dot_prod: malloc failed for result\n");
        return NULL;
    }

    // ------------------------------------------------------------------
    // Compute
    // ------------------------------------------------------------------
    if (use_gpu) {
        runMatMulOp(t1->tensor, t2->tensor, result_array,
                    rows_out, inner, cols_out);
    } else {
        for (int r = 0; r < rows_out; r++) {
            for (int c = 0; c < cols_out; c++) {
                double acc = 0.0;
                for (int k = 0; k < inner; k++)
                    acc += t1->tensor[r * inner + k] * t2->tensor[k * cols_out + c];
                result_array[r * cols_out + c] = acc;
            }
        }
    }
    result_array[out_size] = E;

    // ------------------------------------------------------------------
    // Build shape array (sentinel-terminated)
    // ------------------------------------------------------------------
    // out_ndim is 1 or 2; shape values depend on the case
    int *shape_terminated = (int*) malloc(sizeof(int) * (out_ndim + 1));
    if (!shape_terminated) {
        free(result_array);
        fprintf(stderr, "dot_prod: malloc failed for shape\n");
        return NULL;
    }

    if (out_ndim == 1) {
        // 1D×1D → [1], 1D×2D → [cols], 2D×1D → [rows]
        if (t1->ndim == 1 && t2->ndim == 1)
            shape_terminated[0] = 1;
        else if (t1->ndim == 1 && t2->ndim == 2)
            shape_terminated[0] = cols_out;
        else  // 2D×1D
            shape_terminated[0] = rows_out;
    } else {
        shape_terminated[0] = rows_out;
        shape_terminated[1] = cols_out;
    }
    shape_terminated[out_ndim] = N;

    TensorEngine *result = tensor(result_array, shape_terminated, use_gpu);
    free(result_array);
    free(shape_terminated);
    return result;
}