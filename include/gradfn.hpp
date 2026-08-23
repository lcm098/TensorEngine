#ifndef TENSOR_ENGINE_GRADFN_HPP
#define TENSOR_ENGINE_GRADFN_HPP

#include "common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct TensorEngine;
struct GradFn;

// ============================================================================
// Operation Types in the Dynamic Computational Graph
// ============================================================================

typedef enum GradFnType {
    OP_LEAF = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    OP_ADD_BROAD,
    OP_SUB_BROAD,
    OP_MUL_BROAD,
    OP_DIV_BROAD,
    OP_MOD_BROAD,
    OP_DOT,
    OP_TRANSPOSE,
    OP_RESHAPE,
    OP_SLICE,
    OP_CUSTOM,
    OP_COS,
    OP_SIN,
    OP_CLAMP,
} GradFnType;

// ============================================================================
// Computational Graph Node (GradFn)
// ============================================================================

typedef struct GradFn {
    const char *name;                                                   // Operation name (e.g. "AddBackward")
    GradFnType type;                                                    // Operation identifier enum
    void (*apply)(struct GradFn *self, struct TensorEngine *grad_out);  // Backward backward function callback
    struct TensorEngine **saved_tensors;                               // Saved input tensor references
    size_t num_saved;                                                   // Number of saved input tensors
    void *context;                                                      // Optional operation-specific context (e.g. axes, slice bounds)
    void (*release_context)(void *context);                             // Context cleanup callback
} GradFn;

// ============================================================================
// Core Autograd User API
// ============================================================================

// Configure whether a tensor tracks gradients (leaf node flag)
void set_requires_grad(struct TensorEngine *t, bool requires_grad);

// Reset / zero out accumulated gradients on a tensor
void zero_grad(struct TensorEngine *t);

// Print the DAG computation graph rooted at the given tensor
void print_graph(struct TensorEngine *t);

// Execute the backward pass using reverse topological sort traversal

void backward(struct TensorEngine *root, struct TensorEngine *grad_output);


// ============================================================================
// Internal Graph Construction & Helper API
// ============================================================================

// Allocate and initialize a new GradFn node
GradFn* create_grad_fn(
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, struct TensorEngine *grad_output),
    size_t num_saved,
    struct TensorEngine **saved_tensors,
    void *context,
    void (*release_context)(void *context)
);

// Release memory associated with a GradFn node
void free_grad_fn(GradFn *fn);

// Accumulate incoming gradient into target->grad (handles initialization and shape matching)
void accumulate_grad(struct TensorEngine *target, struct TensorEngine *incoming);

// Sum-reduce gradient along broadcasted axes to match target shape
struct TensorEngine* unbroadcast_to(struct TensorEngine *grad, const int *target_shape, size_t target_ndim);

// ============================================================================
// Backward Functions (apply callbacks for each operation)
// ============================================================================

void add_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void sub_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void mul_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void div_backward_fn(GradFn *self, struct TensorEngine *grad_output);

void add_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void sub_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void mul_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void div_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);

void dot_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void transpose_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void reshape_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void slice_backward_fn(GradFn *self, struct TensorEngine *grad_output);

void _cos_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void _sin_backward_fn(GradFn *self, struct TensorEngine *grad_output);
void _clamp_backward_fn(GradFn *self, struct TensorEngine *grad_output);

// ============================================================================
// Graph Attachment Helpers (called by forward operators)
// ============================================================================

void attach_binary_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *t1,
    struct TensorEngine *t2,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, struct TensorEngine *grad_output)
);

void attach_unary_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, struct TensorEngine *grad_output),
    void *context,
    void (*release_context)(void *ctx)
);

void attach_transpose_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const int *axes
);

void attach_reshape_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const int *orig_shape,
    size_t orig_ndim
);

void attach_slice_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const int *starts,
    const int *steps,
    const int *out_shape,
    int offset
);

void attach_clamp_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    f64 low,
    f64 high
);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_ENGINE_GRADFN_HPP
