
#include "../include/encoder.hpp"


TensorEngine* _encode_image_as_tensor(uint8_t *image_array, size_t H, size_t W, size_t C, bool __GPU_) {
    
    size_t total = H * W * C;
    TensorEngine* t1 = (TensorEngine*)malloc(sizeof(TensorEngine));

    if(t1 == nullptr) {
        fprintf(stderr, "Can not allocated memeory for new image Tensor");
        return NULL;
    }

    t1->tensor = (f64*) malloc(sizeof(f64)*(total+1));

    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            for (size_t c = 0; c < C; c++) {

                size_t index = (h * W * C) + (w * C) + c;
                t1->tensor[index] = (f64)image_array[index];
            }
        }
    }

    t1->tensor[total] = E;

    t1->shape = (int*)malloc(sizeof(int) * 4);
    if (t1->shape == nullptr) {
        free(t1->tensor);
        free(t1);

        fprintf(stderr, "Can not allocate memory for tensor shape\n");
        return NULL;
    }

    t1->shape[0] = H;
    t1->shape[1] = W;
    t1->shape[2] = C;
    t1->shape[3] = N;

    t1->ndim = calculate_ndim(t1->shape);
    t1->size = total;
    t1->strides = calculate_strides(t1->shape, t1->ndim);
    t1->__GPU__ = __GPU_;
    t1->grad = nullptr;
    t1->requires_grad = true;
    t1->is_leaf = true;
    t1->grad_fn = nullptr;
    memset(&t1->gradfn, 0, sizeof(GradFn));
    return t1;
}