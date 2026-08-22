
#ifndef TENSOR_ENGINE_TENSOR_FACTORY_HPP
#define TENSOR_ENGINE_TENSOR_FACTORY_HPP

#include "./common.hpp"
#include "utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif


TensorEngine* ones(int shape[], bool __GPU__);
TensorEngine* zeros(int shape[], bool __GPU__);
TensorEngine* empty(int shape[], bool __GPU__);
TensorEngine* full(int shape[], f64 which, bool __GPU__);

TensorEngine* ones_alike(TensorEngine* t1, bool __GPU__);
TensorEngine* zeros_alike(TensorEngine* t1, bool __GPU__);
TensorEngine* empty_alike(TensorEngine* t1, bool __GPU__);
TensorEngine* full_alike(TensorEngine* t1, f64 which, bool __GPU__);


#ifdef __cplusplus
}
#endif
#endif