
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