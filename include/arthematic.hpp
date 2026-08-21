// include/arthematic.hpp
#ifndef TENSOR_ENGINE_ARTEMATIC_HPP
#define TENSOR_ENGINE_ARTEMATIC_HPP

#include "utils.hpp"
#include "common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// element-wise (matching shapes only)
TensorEngine* add (TensorEngine* t1, TensorEngine* t2);
TensorEngine* sub (TensorEngine* t1, TensorEngine* t2);
TensorEngine* mlt (TensorEngine* t1, TensorEngine* t2);
TensorEngine* divt(TensorEngine* t1, TensorEngine* t2);
TensorEngine* mod (TensorEngine* t1, TensorEngine* t2);

// broadcast
TensorEngine* add_broad(TensorEngine* t1, TensorEngine* t2);
TensorEngine* sub_broad(TensorEngine* t1, TensorEngine* t2);
TensorEngine* mlt_broad(TensorEngine* t1, TensorEngine* t2);
TensorEngine* div_broad(TensorEngine* t1, TensorEngine* t2);
TensorEngine* mod_broad(TensorEngine* t1, TensorEngine* t2);

TensorEngine* dot_prod(TensorEngine* t1, TensorEngine* t2);

#ifdef __cplusplus
}
#endif

#endif