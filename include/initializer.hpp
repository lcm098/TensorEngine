#ifndef TENSOR_ENGINE_INITIALIZER_HPP
#define TENSOR_ENGINE_INITIALIZER_HPP

#include "./utils.hpp"
#include "./common.hpp"

typedef enum {
    INIT_XAVIER_UNIFORM,
    INIT_XAVIER_NORMAL,
    INIT_GLOROT_UNIFORM,
    INIT_GLOROT_NORMAL,
    INIT_KAIMING_UNIFORM,
    INIT_KAIMING_NORMAL,
    INIT_HE_UNIFORM,
    INIT_HE_NORMAL,
    INIT_LECUN_UNIFORM,
    INIT_LECUN_NORMAL
} InitType;

typedef enum {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    GELU,
    Swish,
    Linear
} ActivationType;

typedef enum {
    BIAS_ZEROS,
    BIAS_ONES,
    BIAS_CONSTANT,
    BIAS_UNIFORM,
    BIAS_NORMAL,
    BIAS_TRUNCATED_NORMAL
} BiasInitType;

#ifdef __cplusplus
extern "C" {
#endif

TensorEngine* initialize_tensor(int shape[], InitType type, bool __GPU__);
TensorEngine* initialize_bias(int shape[], BiasInitType type, f64 param1, f64 param2, bool __GPU__);


#ifdef __cplusplus
}
#endif
#endif