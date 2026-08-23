
#include "../include/optimizer.hpp"


OptState* optstate_create(size_t size) {
    OptState *s = (OptState*) calloc(1, sizeof(OptState));
    if (!s) return NULL;
    s->velocity    = (f64*) calloc(size, sizeof(f64));
    s->velocity_sq = (f64*) calloc(size, sizeof(f64));
    s->extra       = (f64*) calloc(size, sizeof(f64));
    s->size = size;
    s->t = 0;
    return s;
}

void optstate_free(OptState *s) {
    if (!s) return;
    free(s->velocity);
    free(s->velocity_sq);
    free(s->extra);
    free(s);
}

// ============================================================================
// Shared helpers
// ============================================================================

static bool fetch_param_grad(TensorEngine *param, f64 **p_host, f64 **g_host) {
    if (param == NULL || param->grad == NULL) return false;
    size_t sz = param->size;

    *p_host = (f64*) malloc(sizeof(f64) * sz);
    *g_host = (f64*) malloc(sizeof(f64) * sz);
    if (!*p_host || !*g_host) {
        free(*p_host); free(*g_host);
        return false;
    }

    if (param->__GPU__) {
        copyDeviceToHost(*p_host, param->tensor, sz);
        copyDeviceToHost(*g_host, param->grad->tensor, sz);
    } else {
        memcpy(*p_host, param->tensor, sizeof(f64) * sz);
        memcpy(*g_host, param->grad->tensor, sizeof(f64) * sz);
    }
    return true;
}

static TensorEngine* build_updated_param(TensorEngine *param, f64 *new_data_no_sentinel) {
    size_t sz = param->size;

    f64 *new_data = (f64*) malloc(sizeof(f64) * (sz + 1));
    if (!new_data) return NULL;
    memcpy(new_data, new_data_no_sentinel, sizeof(f64) * sz);
    new_data[sz] = E;

    int *sh = (int*) malloc(sizeof(int) * (param->ndim + 1));
    if (!sh) { free(new_data); return NULL; }
    for (size_t i = 0; i < param->ndim; i++) sh[i] = param->shape[i];
    sh[param->ndim] = N;

    TensorEngine *updated = tensor(new_data, sh, param->__GPU__);
    free(new_data);
    free(sh);

    if (updated) {
        updated->requires_grad = true;
        updated->is_leaf = true;
        updated->grad = NULL;
        updated->grad_fn = NULL;
    }
    return updated;
}

static bool validate_state(OptState *state, size_t expected_size, const char *fn_name) {
    if (state == NULL) {
        fprintf(stderr, "%s: OptState must not be NULL\n", fn_name);
        return false;
    }
    if (state->size != expected_size) {
        fprintf(stderr, "%s: OptState size (%zu) does not match parameter size (%zu)\n",
                fn_name, state->size, expected_size);
        return false;
    }
    return true;
}

// ============================================================================
// 1. SGD
// ============================================================================
// theta = theta - lr * grad

TensorEngine* sgd_update(TensorEngine *param, f64 lr) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        new_data[i] = p_host[i] - lr * g_host[i];
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 2. Momentum (classic / heavy-ball)
// ============================================================================
// v = momentum * v + grad
// theta = theta - lr * v

TensorEngine* momentum_update(TensorEngine *param, OptState *state, f64 lr, f64 momentum) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "momentum_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        state->velocity[i] = momentum * state->velocity[i] + g_host[i];
        new_data[i] = p_host[i] - lr * state->velocity[i];
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 3. Nesterov Accelerated Gradient
// ============================================================================
// v = momentum * v + grad
// theta = theta - lr * (momentum * v + grad)      [lookahead correction]

TensorEngine* nesterov_update(TensorEngine *param, OptState *state, f64 lr, f64 momentum) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "nesterov_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        f64 v_prev = state->velocity[i];
        state->velocity[i] = momentum * v_prev + g_host[i];
        // Nesterov lookahead: use momentum*v_new + grad, not just v_new
        f64 update = momentum * state->velocity[i] + g_host[i];
        new_data[i] = p_host[i] - lr * update;
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 4. Adagrad
// ============================================================================
// G = G + grad^2
// theta = theta - lr * grad / (sqrt(G) + epsilon)

TensorEngine* adagrad_update(TensorEngine *param, OptState *state, f64 lr, f64 epsilon) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "adagrad_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        state->velocity_sq[i] += g_host[i] * g_host[i];
        new_data[i] = p_host[i] - lr * g_host[i] / (sqrt(state->velocity_sq[i]) + epsilon);
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 5. RMSProp
// ============================================================================
// E[g^2] = decay * E[g^2] + (1-decay) * grad^2
// theta = theta - lr * grad / (sqrt(E[g^2]) + epsilon)

TensorEngine* rmsprop_update(TensorEngine *param, OptState *state, f64 lr, f64 decay, f64 epsilon) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "rmsprop_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        state->velocity_sq[i] = decay * state->velocity_sq[i] + (1.0 - decay) * g_host[i] * g_host[i];
        new_data[i] = p_host[i] - lr * g_host[i] / (sqrt(state->velocity_sq[i]) + epsilon);
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 6. Adam
// ============================================================================
// m = beta1*m + (1-beta1)*grad
// v = beta2*v + (1-beta2)*grad^2
// m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
// theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)

TensorEngine* adam_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "adam_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    state->t += 1;
    f64 bias_correction1 = 1.0 - pow(beta1, state->t);
    f64 bias_correction2 = 1.0 - pow(beta2, state->t);

    for (size_t i = 0; i < sz; i++) {
        state->velocity[i]    = beta1 * state->velocity[i]    + (1.0 - beta1) * g_host[i];
        state->velocity_sq[i] = beta2 * state->velocity_sq[i] + (1.0 - beta2) * g_host[i] * g_host[i];

        f64 m_hat = state->velocity[i] / bias_correction1;
        f64 v_hat = state->velocity_sq[i] / bias_correction2;

        new_data[i] = p_host[i] - lr * m_hat / (sqrt(v_hat) + epsilon);
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 7. AdamW (Adam with decoupled weight decay)
// ============================================================================
// Same as Adam, but weight decay is applied directly to theta,
// NOT folded into the gradient before the moment updates.

TensorEngine* adamw_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon, f64 weight_decay) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "adamw_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    state->t += 1;
    f64 bias_correction1 = 1.0 - pow(beta1, state->t);
    f64 bias_correction2 = 1.0 - pow(beta2, state->t);

    for (size_t i = 0; i < sz; i++) {
        state->velocity[i]    = beta1 * state->velocity[i]    + (1.0 - beta1) * g_host[i];
        state->velocity_sq[i] = beta2 * state->velocity_sq[i] + (1.0 - beta2) * g_host[i] * g_host[i];

        f64 m_hat = state->velocity[i] / bias_correction1;
        f64 v_hat = state->velocity_sq[i] / bias_correction2;

        f64 adam_step = lr * m_hat / (sqrt(v_hat) + epsilon);
        f64 decay_step = lr * weight_decay * p_host[i];

        new_data[i] = p_host[i] - adam_step - decay_step;
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 8. Adamax (Adam variant using infinity norm)
// ============================================================================
// m = beta1*m + (1-beta1)*grad
// u = max(beta2*u, |grad|)
// theta = theta - (lr / (1 - beta1^t)) * m / (u + epsilon)

TensorEngine* adamax_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "adamax_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    state->t += 1;
    f64 bias_correction1 = 1.0 - pow(beta1, state->t);

    for (size_t i = 0; i < sz; i++) {
        state->velocity[i] = beta1 * state->velocity[i] + (1.0 - beta1) * g_host[i];

        f64 abs_grad = fabs(g_host[i]);
        f64 scaled_u = beta2 * state->velocity_sq[i];
        state->velocity_sq[i] = (scaled_u > abs_grad) ? scaled_u : abs_grad; // infinity norm (max)

        new_data[i] = p_host[i] - (lr / bias_correction1) * state->velocity[i] / (state->velocity_sq[i] + epsilon);
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 9. Nadam (Adam + Nesterov momentum)
// ============================================================================
// m = beta1*m + (1-beta1)*grad
// v = beta2*v + (1-beta2)*grad^2
// m_hat = beta1*m/(1-beta1^(t+1)) + (1-beta1)*grad/(1-beta1^t)   [Nesterov lookahead]
// v_hat = v / (1 - beta2^t)
// theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)

TensorEngine* nadam_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 epsilon) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "nadam_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    state->t += 1;
    f64 bias_correction1     = 1.0 - pow(beta1, state->t);
    f64 bias_correction1_next = 1.0 - pow(beta1, state->t + 1);
    f64 bias_correction2     = 1.0 - pow(beta2, state->t);

    for (size_t i = 0; i < sz; i++) {
        state->velocity[i]    = beta1 * state->velocity[i]    + (1.0 - beta1) * g_host[i];
        state->velocity_sq[i] = beta2 * state->velocity_sq[i] + (1.0 - beta2) * g_host[i] * g_host[i];

        f64 m_hat = (beta1 * state->velocity[i] / bias_correction1_next)
                  + ((1.0 - beta1) * g_host[i] / bias_correction1);
        f64 v_hat = state->velocity_sq[i] / bias_correction2;

        new_data[i] = p_host[i] - lr * m_hat / (sqrt(v_hat) + epsilon);
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 10. Adadelta
// ============================================================================
// E[g^2] = rho * E[g^2] + (1-rho) * grad^2
// delta = -sqrt(E[dx^2] + epsilon) / sqrt(E[g^2] + epsilon) * grad
// E[dx^2] = rho * E[dx^2] + (1-rho) * delta^2
// theta = theta + delta
// (no explicit learning rate; scale is self-adapting)

TensorEngine* adadelta_update(TensorEngine *param, OptState *state, f64 rho, f64 epsilon) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "adadelta_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        // velocity_sq holds E[g^2], extra holds E[dx^2]
        state->velocity_sq[i] = rho * state->velocity_sq[i] + (1.0 - rho) * g_host[i] * g_host[i];

        f64 delta = -(sqrt(state->extra[i] + epsilon) / sqrt(state->velocity_sq[i] + epsilon)) * g_host[i];

        state->extra[i] = rho * state->extra[i] + (1.0 - rho) * delta * delta;

        new_data[i] = p_host[i] + delta;
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 11. FTRL (Follow The Regularized Leader)
// ============================================================================
// z = z + grad - sigma * theta      where sigma = (sqrt(n_new) - sqrt(n_old)) / lr
// n = n + grad^2
// theta = -sign(z)/((beta+sqrt(n))/lr + l2) * max(0, |z| - l1)     when |z| > l1, else 0

TensorEngine* ftrl_update(TensorEngine *param, OptState *state, f64 lr, f64 lr_power, f64 l1, f64 l2) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "ftrl_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    const f64 beta = 1.0; // standard FTRL beta smoothing term

    for (size_t i = 0; i < sz; i++) {
        // velocity_sq holds n (accumulated squared grads), extra holds z
        f64 n_old = state->velocity_sq[i];
        f64 n_new = n_old + g_host[i] * g_host[i];

        f64 sigma = (pow(n_new, -lr_power) - pow(n_old, -lr_power)) / lr;

        state->extra[i] += g_host[i] - sigma * p_host[i];
        state->velocity_sq[i] = n_new;

        f64 z = state->extra[i];

        if (fabs(z) <= l1) {
            new_data[i] = 0.0;
        } else {
            f64 sign_z = (z > 0.0) ? 1.0 : -1.0;
            f64 denom = (beta + sqrt(n_new)) / lr + l2;
            new_data[i] = -(z - sign_z * l1) / denom;
        }
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}

// ============================================================================
// 12. Lion (EvoLved Sign Momentum)
// ============================================================================
// c = beta1*m + (1-beta1)*grad          [interpolated update direction]
// theta = theta - lr * (sign(c) + weight_decay * theta)
// m = beta2*m + (1-beta2)*grad          [momentum update, AFTER using old m for c]

TensorEngine* lion_update(TensorEngine *param, OptState *state, f64 lr, f64 beta1, f64 beta2, f64 weight_decay) {
    f64 *p_host, *g_host;
    if (!fetch_param_grad(param, &p_host, &g_host)) return NULL;
    if (!validate_state(state, param->size, "lion_update")) { free(p_host); free(g_host); return NULL; }

    size_t sz = param->size;
    f64 *new_data = (f64*) malloc(sizeof(f64) * sz);

    for (size_t i = 0; i < sz; i++) {
        f64 c = beta1 * state->velocity[i] + (1.0 - beta1) * g_host[i];
        f64 sign_c = (c > 0.0) ? 1.0 : ((c < 0.0) ? -1.0 : 0.0);

        new_data[i] = p_host[i] - lr * (sign_c + weight_decay * p_host[i]);

        // momentum update happens AFTER computing c, using the ORIGINAL m
        state->velocity[i] = beta2 * state->velocity[i] + (1.0 - beta2) * g_host[i];
    }

    TensorEngine *updated = build_updated_param(param, new_data);
    free(p_host); free(g_host); free(new_data);
    return updated;
}