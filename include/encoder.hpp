
#ifndef TENSOR_ENGINE_ENCODER_HPP
#define TENSOR_ENGINE_ENCODER_HPP

#include "utils.hpp"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif


TensorEngine* _encode_image_as_tensor(uint8_t *image_array, size_t H, size_t W, size_t C, bool __GPU__);


#ifdef __cplusplus
}
#endif
#endif