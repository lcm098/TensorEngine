#include "../include/pipeline.hpp"

__pipeLine__* build(Layer *first, ...) {
    if (first == NULL) {
        fprintf(stderr, "build: pipeline must contain at least one layer\n");
        return NULL;
    }

    size_t count = 1;
    va_list args;
    va_start(args, first);

    Layer *cur = va_arg(args, Layer*);
    while (cur != NULL) {
        count++;
        cur = va_arg(args, Layer*);
    }
    va_end(args);

    __pipeLine__ *p = (__pipeLine__*) malloc(sizeof(__pipeLine__));
    if (p == NULL) {
        fprintf(stderr, "build: memory allocation failed for pipeline\n");
        return NULL;
    }

    p->layers = (Layer**) malloc(sizeof(Layer*) * count);
    if (p->layers == NULL) {
        fprintf(stderr, "build: memory allocation failed for layer array\n");
        free(p);
        return NULL;
    }

    p->layers[0] = first;

    va_start(args, first);
    for (size_t i = 1; i < count; i++) {
        p->layers[i] = va_arg(args, Layer*);
    }
    va_end(args);

    p->depth = count;
    return p;
}

static void print_tensor_info(const char *label, TensorEngine *t) {
    if (t == NULL) {
        printf("%s: invalid tensor\n", label);
        return;
    }

    printf("%s: shape = [", label);
    for (size_t d = 0; d < t->ndim; d++) {
        printf("%d", t->shape[d]);
        if (d + 1 < t->ndim) printf(", ");
    }
    printf("], size = %zu, __gpu__ = %s\n", t->size, t->__GPU__ ? "true" : "false");
}

static const char *map_activation(int activation_no) {
    switch (activation_no) {
        case Sigmoid:
            return "Sigmoid";
        case ReLU:
            return "ReLU";
        case Tanh:
            return "Tanh";
        case Softmax:
            return "Softmax";
        case LeakyReLU:
            return "LeakyReLU";
        case ELU:
            return "ELU";
        case GELU:
            return "GELU";
        case Swish:
            return "Swish";
        case Linear:
            return "Linear";
        default:
            return "Invalid Activation";
    }
}

void print_pipeline(__pipeLine__ *pipe) {
    if (pipe == NULL) {
        fprintf(stderr, "print_pipeline: null pipeline argument\n");
        return;
    }

    printf("[################ PIPELINE %zu ###################]\n", pipe->depth);

    for (size_t i = 0; i < pipe->depth; i++) {
        printf("=============== Layer %zu ===============\n", i);

        Layer *layer = pipe->layers[i];

        if (layer == NULL) {
            fprintf(stderr, "invalid layer\n");
            continue;
        }

        printf("in_feature = %d, out_feature = %d, activation = %s\n",
               layer->in_feature, layer->out_feature, map_activation((int) layer->activation));

        print_tensor_info("weights", layer->weights);
        print_tensor_info("bias", layer->bias);
    }

    printf("[#################################################]\n");
}

void free_pipeline(__pipeLine__ *p) {
    if (p == NULL) return;

    for (size_t i = 0; i < p->depth; i++) {
        free_layer(p->layers[i]);
    }

    free(p->layers);
    free(p);
}