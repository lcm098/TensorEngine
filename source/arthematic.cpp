
#include "../include/arthematic.hpp"
#include "../include/common.hpp"
#include "../__cuda__/include/kernel.cuh"

TensorEngine* add(TensorEngine* t1, TensorEngine* t2) {
    if (t1->size != t2->size) {
        fprintf(stderr, "Tensor SIZE are not balanced\n");
        return NULL;
    }

    if (t1->__GPU__ && t2->__GPU__) {
        // +1 slot for the E (-INFINITY) sentinel calculate_elements() scans for
        
        double *h_cd = (double *)malloc((t1->size + 1) * sizeof(double));
        runDoubleOp(addKernelD, t1->tensor, t2->tensor, h_cd, t1->size);
        h_cd[t1->size] = E;

        int *shape_terminated = (int*) malloc(sizeof(int) * (t1->ndim + 1));
        for (size_t i = 0; i < t1->ndim; i++) {
            shape_terminated[i] = t1->shape[i];
        }
        shape_terminated[t1->ndim] = N;

        TensorEngine *result = tensor(h_cd, shape_terminated, true);

        free(shape_terminated);
        free(h_cd);

        return result;
    }

    fprintf(stderr, "CPU path not implemented\n");
    return NULL;
}