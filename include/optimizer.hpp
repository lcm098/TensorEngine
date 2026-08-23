#ifndef TENSOR_ENGINE_OPTIMIZER_HPP
#define TENSOR_ENGINE_OPTIMIZER_HPP

#include "common.hpp"
#include "utils.hpp"
#include "../__cuda__/include/kernel.cuh"

// Holds per-parameter optimizer state needed by stateful optimizers
// (Momentum, Adam, RMSProp, etc.). One OptState should be created per
// trainable tensor and kept alive across training steps.

typedef struct {
    f64 *velocity;      // 1st moment / momentum buffer
    f64 *velocity_sq;   // 2nd moment / squared-gradient buffer
    f64 *extra;         // 3rd buffer, used by Nadam (momentum lookahead) / Adadelta (accum update) / FTRL (z)
    size_t size;
    int t;               // timestep, used for bias correction (Adam family)
} OptState;

#ifdef __cplusplus
extern "C" {
#endif

OptState* optstate_create(size_t size);
void optstate_free(OptState *s);

TensorEngine* sgd_update(TensorEngine *param, f64 lr);
TensorEngine* momentum_update(TensorEngine *param, OptState *state, f64 lr, f64 momentum);
TensorEngine* nesterov_update(TensorEngine *param, OptState *state, f64 lr, f64 momentum);
TensorEngine* adagrad_update(TensorEngine *param, OptState *state, f64 lr, f64 epsilon);
TensorEngine* rmsprop_update(TensorEngine *param, OptState *state, f64 lr, f64 decay, f64 epsilon);
TensorEngine* adam_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon);
TensorEngine* adamw_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon, f64 weight_decay);
TensorEngine* adamax_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon);
TensorEngine* nadam_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon);
TensorEngine* adadelta_update(TensorEngine *param, OptState *state, f64 rho, f64 epsilon);
TensorEngine* ftrl_update(TensorEngine *param, OptState *state, f64 lr, f64 lr_power, f64 l1, f64 l2);
TensorEngine* lion_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 weight_decay);

#ifdef __cplusplus
}
#endif
#endif