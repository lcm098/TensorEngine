#include "../include/pipeline.hpp"

__pipeLine__* build(TensorEngine *first, ...) {
    if (first == NULL) {
        fprintf(stderr, "build: pipeline must contain at least one layer\n");
        return NULL;
    }

    /* first pass: count how many non-NULL args were given */
    size_t count = 1;
    va_list args;
    va_start(args, first);

    TensorEngine *cur = va_arg(args, TensorEngine*);
    while (cur != NULL) {
        count++;
        cur = va_arg(args, TensorEngine*);
    }
    va_end(args);

    __pipeLine__ *p = (__pipeLine__*) malloc(sizeof(__pipeLine__));
    if (p == NULL) {
        fprintf(stderr, "build: memory allocation failed for pipeline\n");
        return NULL;
    }

    p->layers = (TensorEngine**) malloc(sizeof(TensorEngine*) * count);
    if (p->layers == NULL) {
        fprintf(stderr, "build: memory allocation failed for layer array\n");
        free(p);
        return NULL;
    }

    /* second pass: actually store the pointers */
    p->layers[0] = first;

    va_start(args, first);
    for (size_t i = 1; i < count; i++) {
        p->layers[i] = va_arg(args, TensorEngine*);
    }
    va_end(args);

    p->depth = count;
    return p;
}

void print_pipeline(__pipeLine__ *pipe) {
    if (pipe == NULL) {
        fprintf(stderr, "print_pipeline: null pipeline argument\n");
        return;
    }

    printf("########## PIPELINE (%zu layers) ##########\n", pipe->depth);

    for (size_t i = 0; i < pipe->depth; i++) {
        printf("---- Layer %zu ----\n", i);

        TensorEngine *t = pipe->layers[i];

        if (t == NULL) {
            fprintf(stderr, "invalid tensor\n");
            continue;
        }

        printf("size = %zu\n", t->size);
        printf("ndim = %zu\n", t->ndim);

        printf("shape = [");
        for (size_t d = 0; d < t->ndim; d++) {
            printf("%d", t->shape[d]);
            if (d + 1 < t->ndim) printf(", ");
        }
        printf("]\n");

        printf("strides = [");
        for (size_t d = 0; d < t->ndim; d++) {
            printf("%d", t->strides[d]);
            if (d + 1 < t->ndim) printf(", ");
        }
        printf("]\n");

        const char *status = t->__GPU__ ? "true" : "false";
        printf("__gpu__ [%s]\n", status);
    }

    printf("############################################\n");
}

void free_pipeline(__pipeLine__ *p) {
    if (p == NULL) return;

    for (size_t i = 0; i < p->depth; i++) {
        TensorEngine *t = p->layers[i];
        if (t == NULL) continue;

        if (t->__GPU__) {
            freeDeviceMemory(t->tensor);
        } else {
            free(t->tensor);
        }

        free(t->shape);
        free(t->strides);
        free(t);
    }

    free(p->layers);
    free(p);
}