#include "../include/tensor_factory.hpp"


int compute_size(int shape[]) {
    int product = 1;
    int track = 0;

    while (shape[track] != N) {
        product *= shape[track];
        track++;
    }

    return product;
}


static TensorEngine* alloc_filled(int shape[], f64 fill_value, bool __GPU__, bool skip_fill) {
    int total_ele = compute_size(shape);

    TensorEngine* t1 = (TensorEngine*) malloc(sizeof(TensorEngine));
    if (t1 == nullptr) {
        fprintf(stderr, "Memory allocation for new tensor, got failed\n");
        return NULL;
    }

    // count dims first, so we know how much to allocate/copy for shape
    int ndim = 0;
    while (shape[ndim] != N) {
        ndim++;
    }

    t1->shape = (int*) malloc(sizeof(int) * (ndim + 1));
    if (t1->shape == nullptr) {
        fprintf(stderr, "Memory allocation for tensor shape, got failed\n");
        free(t1);
        return NULL;
    }
    for (int i = 0; i <= ndim; i++) {
        t1->shape[i] = shape[i]; // copies the N terminator too
    }

    t1->tensor = (f64*) malloc(sizeof(f64) * (total_ele + 1));
    if (t1->tensor == nullptr) {
        fprintf(stderr, "Memory allocation for tensor data, got failed\n");
        free(t1->shape);
        free(t1);
        return NULL;
    }

    if (!skip_fill) {
        for (int i = 0; i < total_ele; i++) {
            t1->tensor[i] = fill_value;
        }
    }
    t1->tensor[total_ele] = E;

    t1->ndim = calculate_ndim(t1->shape);
    t1->size = total_ele;
    t1->strides = calculate_strides(t1->shape, t1->ndim);
    t1->__GPU__ = __GPU__;

    return t1;
}

TensorEngine* zeros(int shape[], bool __GPU__) {
    return alloc_filled(shape, 0.0, __GPU__, false);
}

TensorEngine* empty(int shape[], bool __GPU__) {
    return alloc_filled(shape, 0.0, __GPU__, true); // fill_value ignored when skip_fill=true
}

TensorEngine* full(int shape[], f64 which, bool __GPU__) {
    return alloc_filled(shape, which, __GPU__, false);
}

TensorEngine* ones(int shape[], bool __GPU__) {
    return alloc_filled(shape, 1.0, __GPU__, false);
}

TensorEngine* ones_alike(TensorEngine* t1, bool __GPU__) {
    if (t1 == nullptr) {
        fprintf(stderr, "ones_alike: null tensor argument\n");
        return NULL;
    }
    return alloc_filled(t1->shape, 1.0, __GPU__, false);
}

TensorEngine* zeros_alike(TensorEngine* t1, bool __GPU__) {
    if (t1 == nullptr) {
        fprintf(stderr, "zeros_alike: null tensor argument\n");
        return NULL;
    }
    return alloc_filled(t1->shape, 0.0, __GPU__, false);
}

TensorEngine* empty_alike(TensorEngine* t1, bool __GPU__) {
    if (t1 == nullptr) {
        fprintf(stderr, "empty_alike: null tensor argument\n");
        return NULL;
    }
    return alloc_filled(t1->shape, 0.0, __GPU__, true); // fill_value ignored when skip_fill=true
}

TensorEngine* full_alike(TensorEngine* t1, f64 which, bool __GPU__) {
    if (t1 == nullptr) {
        fprintf(stderr, "full_alike: null tensor argument\n");
        return NULL;
    }
    return alloc_filled(t1->shape, which, __GPU__, false);
}