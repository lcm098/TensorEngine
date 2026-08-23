// source/gradfn.cpp
// Dynamic Computational Graph Engine (Autograd)

#include "../include/gradfn.hpp"
#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"
#include "../include/tensor_factory.hpp"
#include "../include/math.hpp"


// ============================================================================
// Internal Helpers
// ============================================================================

// Creates a standalone copy of a tensor's data and shape (no graph attachment)
static TensorEngine* clone_tensor_data(TensorEngine *src) {
    if (!src) return nullptr;

    f64 *data = (f64*) malloc(sizeof(f64) * (src->size + 1));
    if (!data) return nullptr;
    memcpy(data, src->tensor, sizeof(f64) * src->size);
    data[src->size] = E;

    int *sh = (int*) malloc(sizeof(int) * (src->ndim + 1));
    if (!sh) {
        free(data);
        return nullptr;
    }
    for (size_t i = 0; i < src->ndim; i++) {
        sh[i] = src->shape[i];
    }
    sh[src->ndim] = N;

    TensorEngine *out = tensor(data, sh, src->__GPU__);
    free(data);
    free(sh);
    return out;
}

// Creates a scalar-filled tensor matching the shape and device target of src
static TensorEngine* scalar_like(TensorEngine *src, f64 value) {
    int *sh = (int*) malloc(sizeof(int) * (src->ndim + 1));
    if (!sh) return nullptr;

    for (size_t i = 0; i < src->ndim; i++) {
        sh[i] = src->shape[i];
    }
    sh[src->ndim] = N;

    TensorEngine *out = full(sh, value, src->__GPU__);
    free(sh);
    return out;
}

// Negates all elements in a tensor (-1.0 * t)
static TensorEngine* negate_tensor(TensorEngine *t) {
    TensorEngine *neg_one = scalar_like(t, -1.0);
    if (!neg_one) return nullptr;

    TensorEngine *result = mlt(t, neg_one);
    free_tensor(neg_one);
    return result;
}

// Computes element-wise square (t * t)
static TensorEngine* square_tensor(TensorEngine *t) {
    return mlt(t, t);
}

// ============================================================================
// GradFn Lifecycle Management
// ============================================================================

GradFn* create_grad_fn(
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, TensorEngine *grad_output),
    size_t num_saved,
    TensorEngine **saved_tensors,
    void *context,
    void (*release_context)(void *context))
{
    GradFn *fn = (GradFn*) malloc(sizeof(GradFn));
    if (!fn) {
        fprintf(stderr, "create_grad_fn: allocation failed\n");
        return nullptr;
    }

    fn->name            = name;
    fn->type            = type;
    fn->apply           = apply;
    fn->num_saved       = num_saved;
    fn->context         = context;
    fn->release_context = release_context;

    if (num_saved > 0 && saved_tensors != nullptr) {
        fn->saved_tensors = (TensorEngine**) malloc(sizeof(TensorEngine*) * num_saved);
        if (!fn->saved_tensors) {
            fprintf(stderr, "create_grad_fn: saved_tensors allocation failed\n");
            free(fn);
            return nullptr;
        }
        for (size_t i = 0; i < num_saved; i++) {
            fn->saved_tensors[i] = saved_tensors[i];
        }
    } else {
        fn->saved_tensors = nullptr;
    }

    return fn;
}

void free_grad_fn(GradFn *fn) {
    if (!fn) return;

    if (fn->release_context && fn->context) {
        fn->release_context(fn->context);
    }
    free(fn->saved_tensors);
    free(fn);
}

// ============================================================================
// Gradient Accumulation
// ============================================================================

void accumulate_grad(TensorEngine *target, TensorEngine *incoming) {
    if (!target || !incoming) return;

    if (target->grad == nullptr) {
        target->grad = clone_tensor_data(incoming);
    } else {
        // Check if shapes match exactly
        bool shapes_match = (target->grad->ndim == incoming->ndim);
        if (shapes_match) {
            for (size_t i = 0; i < target->grad->ndim; i++) {
                if (target->grad->shape[i] != incoming->shape[i]) {
                    shapes_match = false;
                    break;
                }
            }
        }

        TensorEngine *new_grad = nullptr;
        if (shapes_match) {
            new_grad = add(target->grad, incoming);
        } else {
            new_grad = add_broad(target->grad, incoming);
        }

        free_tensor(target->grad);
        target->grad = new_grad;
    }
}

// ============================================================================
// Gradient Unbroadcasting (Sum-reduction along broadcasted dimensions)
// ============================================================================

TensorEngine* unbroadcast_to(TensorEngine *grad, const int *target_shape, size_t target_ndim) {
    if (!grad || !target_shape) return nullptr;

    // Fast path: shapes already match identically
    if (grad->ndim == target_ndim) {
        bool match = true;
        for (size_t i = 0; i < target_ndim; i++) {
            if (grad->shape[i] != target_shape[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return clone_tensor_data(grad);
        }
    }

    size_t out_ndim  = target_ndim;
    size_t grad_ndim = grad->ndim;
    size_t pad       = grad_ndim - out_ndim;

    // Compute total target element count
    size_t out_size = 1;
    for (size_t i = 0; i < out_ndim; i++) {
        out_size *= (size_t) target_shape[i];
    }

    f64 *result = (f64*) calloc(out_size + 1, sizeof(f64));
    if (!result) return nullptr;

    size_t *out_strides = (size_t*) malloc(sizeof(size_t) * out_ndim);
    if (!out_strides) {
        free(result);
        return nullptr;
    }

    // Compute row-major strides for target output
    {
        size_t acc = 1;
        for (size_t i = out_ndim; i-- > 0; ) {
            out_strides[i] = acc;
            acc *= (size_t) target_shape[i];
        }
    }

    // Accumulate each element of grad into its reduced target position
    for (size_t linear = 0; linear < grad->size; linear++) {
        size_t rem = linear;
        size_t out_offset = 0;

        for (size_t d = grad_ndim; d-- > 0; ) {
            size_t coord = rem % (size_t) grad->shape[d];
            rem /= (size_t) grad->shape[d];

            if (d >= pad) {
                size_t target_d = d - pad;
                size_t target_coord = (target_shape[target_d] == 1) ? 0 : coord;
                out_offset += target_coord * out_strides[target_d];
            }
        }

        result[out_offset] += grad->tensor[linear];
    }
    result[out_size] = E;

    int *sh = (int*) malloc(sizeof(int) * (out_ndim + 1));
    if (!sh) {
        free(result);
        free(out_strides);
        return nullptr;
    }
    for (size_t i = 0; i < out_ndim; i++) {
        sh[i] = target_shape[i];
    }
    sh[out_ndim] = N;

    TensorEngine *out = tensor(result, sh, grad->__GPU__);
    free(result);
    free(sh);
    free(out_strides);
    return out;
}

// ============================================================================
// Core Autograd API Functions
// ============================================================================

void set_requires_grad(TensorEngine *t, bool requires_grad) {
    if (!t) return;
    t->requires_grad = requires_grad;
    t->is_leaf = true;
}

void zero_grad(TensorEngine *t) {
    if (!t) return;
    if (t->grad != nullptr) {
        free_tensor(t->grad);
        t->grad = nullptr;
    }
}

// ============================================================================
// Backward Pass (Topological Sort + Reverse DAG Traversal)
// ============================================================================

// Dynamic list of nodes for ordered execution
typedef struct {
    TensorEngine **nodes;
    size_t count;
    size_t capacity;
} NodeList;

static void nodelist_init(NodeList *nl) {
    nl->count = 0;
    nl->capacity = 64;
    nl->nodes = (TensorEngine**) malloc(sizeof(TensorEngine*) * nl->capacity);
}

static void nodelist_push(NodeList *nl, TensorEngine *t) {
    if (nl->count >= nl->capacity) {
        nl->capacity *= 2;
        nl->nodes = (TensorEngine**) realloc(nl->nodes, sizeof(TensorEngine*) * nl->capacity);
    }
    nl->nodes[nl->count++] = t;
}

static void nodelist_free(NodeList *nl) {
    free(nl->nodes);
    nl->nodes = nullptr;
    nl->count = 0;
    nl->capacity = 0;
}

// Visited set for cycle prevention and duplicate avoidance
typedef struct {
    TensorEngine **set;
    size_t count;
    size_t capacity;
} VisitedSet;

static void visited_init(VisitedSet *vs) {
    vs->count = 0;
    vs->capacity = 64;
    vs->set = (TensorEngine**) malloc(sizeof(TensorEngine*) * vs->capacity);
}

static bool visited_contains(VisitedSet *vs, TensorEngine *t) {
    for (size_t i = 0; i < vs->count; i++) {
        if (vs->set[i] == t) return true;
    }
    return false;
}

static void visited_add(VisitedSet *vs, TensorEngine *t) {
    if (vs->count >= vs->capacity) {
        vs->capacity *= 2;
        vs->set = (TensorEngine**) realloc(vs->set, sizeof(TensorEngine*) * vs->capacity);
    }
    vs->set[vs->count++] = t;
}

static void visited_free(VisitedSet *vs) {
    free(vs->set);
    vs->set = nullptr;
    vs->count = 0;
    vs->capacity = 0;
}

// Depth-first search for post-order topological sort
static void topo_sort_dfs(TensorEngine *t, VisitedSet *visited, NodeList *order) {
    if (!t || visited_contains(visited, t)) return;
    visited_add(visited, t);

    if (t->grad_fn && t->grad_fn->saved_tensors) {
        for (size_t i = 0; i < t->grad_fn->num_saved; i++) {
            TensorEngine *parent = t->grad_fn->saved_tensors[i];
            if (parent) {
                topo_sort_dfs(parent, visited, order);
            }
        }
    }
    nodelist_push(order, t);
}

void backward(TensorEngine *root, TensorEngine *grad_output) {
    if (!root) {
        fprintf(stderr, "backward: null root tensor\n");
        return;
    }

    // Seed root gradient with 1.0 (matching root tensor shape) if not explicitly provided
    if (grad_output == nullptr) {
        int *sh = (int*) malloc(sizeof(int) * (root->ndim + 1));
        for (size_t i = 0; i < root->ndim; i++) {
            sh[i] = root->shape[i];
        }
        sh[root->ndim] = N;

        grad_output = ones(sh, root->__GPU__);
        free(sh);
        accumulate_grad(root, grad_output);
        free_tensor(grad_output);
    } else {
        accumulate_grad(root, grad_output);
    }

    // Build topological execution order
    NodeList order;
    VisitedSet visited;
    nodelist_init(&order);
    visited_init(&visited);

    topo_sort_dfs(root, &visited, &order);

    // Execute backward functions in reverse topological order (root -> leaves)
    for (size_t i = order.count; i-- > 0; ) {
        TensorEngine *node = order.nodes[i];
        if (!node->grad_fn || !node->grad_fn->apply) continue;
        if (!node->grad) continue;

        node->grad_fn->apply(node->grad_fn, node->grad);
    }

    nodelist_free(&order);
    visited_free(&visited);
}

// ============================================================================
// Computational Graph Visualization
// ============================================================================

static void print_graph_recursive(TensorEngine *t, int depth, VisitedSet *visited) {
    if (!t) return;

    for (int i = 0; i < depth; i++) {
        printf("  ");
    }

    if (t->grad_fn && t->grad_fn->name) {
        printf("-> %s", t->grad_fn->name);
    } else if (t->is_leaf && t->requires_grad) {
        printf("-> [Leaf: requires_grad=true]");
    } else if (t->is_leaf) {
        printf("-> [Leaf]");
    } else {
        printf("-> [Tensor]");
    }

    printf(" | shape=[");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i + 1 < t->ndim) printf(", ");
    }
    printf("]");

    if (t->grad) printf(" | has_grad");
    printf("\n");

    if (visited_contains(visited, t)) return;
    visited_add(visited, t);

    if (t->grad_fn && t->grad_fn->saved_tensors) {
        for (size_t i = 0; i < t->grad_fn->num_saved; i++) {
            print_graph_recursive(t->grad_fn->saved_tensors[i], depth + 1, visited);
        }
    }
}

void print_graph(TensorEngine *t) {
    if (!t) {
        fprintf(stderr, "print_graph: null tensor\n");
        return;
    }
    printf("=== Computation Graph ===\n");
    VisitedSet visited;
    visited_init(&visited);
    print_graph_recursive(t, 0, &visited);
    visited_free(&visited);
    printf("=========================\n");
}

// ============================================================================
// Backward Implementations: Basic Arithmetic (Element-wise)
// ============================================================================

// c = a + b  =>  da = dc,  db = dc
void add_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) accumulate_grad(a, grad_output);
    if (b) accumulate_grad(b, grad_output);
}

// c = a - b  =>  da = dc,  db = -dc
void sub_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) accumulate_grad(a, grad_output);
    if (b) {
        TensorEngine *neg_grad = negate_tensor(grad_output);
        accumulate_grad(b, neg_grad);
        free_tensor(neg_grad);
    }
}

// c = a * b  =>  da = dc * b,  db = dc * a
void mul_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) {
        TensorEngine *da = mlt(grad_output, b);
        accumulate_grad(a, da);
        free_tensor(da);
    }
    if (b) {
        TensorEngine *db = mlt(grad_output, a);
        accumulate_grad(b, db);
        free_tensor(db);
    }
}

// c = a / b  =>  da = dc / b,  db = -dc * a / (b^2)
void div_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) {
        TensorEngine *da = divt(grad_output, b);
        accumulate_grad(a, da);
        free_tensor(da);
    }
    if (b) {
        TensorEngine *b_sq       = square_tensor(b);
        TensorEngine *a_over_bsq = divt(a, b_sq);
        TensorEngine *dc_times   = mlt(grad_output, a_over_bsq);
        TensorEngine *neg        = negate_tensor(dc_times);

        accumulate_grad(b, neg);
        free_tensor(b_sq);
        free_tensor(a_over_bsq);
        free_tensor(dc_times);
        free_tensor(neg);
    }
}

// c = cos(a)  =>  da = -dc * sin(a)
void _cos_backward_fn(GradFn *self, struct TensorEngine *grad_output) {
    if (!self || !grad_output || self->num_saved < 1) return;
    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;

    f64 *da_data = (f64*) malloc(sizeof(f64) * (input->size + 1));
    if (!da_data) return;

    for (size_t i = 0; i < input->size; i++) {
        da_data[i] = grad_output->tensor[i] * (-std::sin(input->tensor[i]));
    }
    da_data[input->size] = E;

    int *sh = (int*) malloc(sizeof(int) * (input->ndim + 1));
    if (!sh) {
        free(da_data);
        return;
    }
    for (size_t i = 0; i < input->ndim; i++) {
        sh[i] = input->shape[i];
    }
    sh[input->ndim] = N;

    TensorEngine *da = tensor(da_data, sh, input->__GPU__);
    free(da_data);
    free(sh);

    if (da) {
        accumulate_grad(input, da);
        free_tensor(da);
    }
}

// c = sin(a)  =>  da = dc * cos(a)
void _sin_backward_fn(GradFn *self, struct TensorEngine *grad_output) {
    if (!self || !grad_output || self->num_saved < 1) return;
    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;

    f64 *da_data = (f64*) malloc(sizeof(f64) * (input->size + 1));
    if (!da_data) return;

    for (size_t i = 0; i < input->size; i++) {
        da_data[i] = grad_output->tensor[i] * std::cos(input->tensor[i]);
    }
    da_data[input->size] = E;

    int *sh = (int*) malloc(sizeof(int) * (input->ndim + 1));
    if (!sh) {
        free(da_data);
        return;
    }
    for (size_t i = 0; i < input->ndim; i++) {
        sh[i] = input->shape[i];
    }
    sh[input->ndim] = N;

    TensorEngine *da = tensor(da_data, sh, input->__GPU__);
    free(da_data);
    free(sh);

    if (da) {
        accumulate_grad(input, da);
        free_tensor(da);
    }
}

// ---------- Clamp Backward ----------
typedef struct {
    f64 low;
    f64 high;
} ClampCtx;

static void release_clamp_ctx(void *context) {
    ClampCtx *ctx = (ClampCtx*) context;
    if (ctx) {
        free(ctx);
    }
}

// c = clamp(a, low, high)  =>  da = dc if low <= a <= high else 0
void _clamp_backward_fn(GradFn *self, struct TensorEngine *grad_output) {
    if (!self || !grad_output || self->num_saved < 1) return;
    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;

    ClampCtx *ctx = (ClampCtx*) self->context;
    f64 low = ctx ? ctx->low : -INFINITY;
    f64 high = ctx ? ctx->high : INFINITY;

    f64 *da_data = (f64*) malloc(sizeof(f64) * (input->size + 1));
    if (!da_data) return;

    for (size_t i = 0; i < input->size; i++) {
        f64 x = input->tensor[i];
        if (x >= low && x <= high) {
            da_data[i] = grad_output->tensor[i];
        } else {
            da_data[i] = 0.0;
        }
    }
    da_data[input->size] = E;

    int *sh = (int*) malloc(sizeof(int) * (input->ndim + 1));
    if (!sh) {
        free(da_data);
        return;
    }
    for (size_t i = 0; i < input->ndim; i++) {
        sh[i] = input->shape[i];
    }
    sh[input->ndim] = N;

    TensorEngine *da = tensor(da_data, sh, input->__GPU__);
    free(da_data);
    free(sh);

    if (da) {
        accumulate_grad(input, da);
        free_tensor(da);
    }
}

// ============================================================================
// Backward Implementations: Broadcast Arithmetic
// ============================================================================

// c = a + b (broadcast)  =>  da = unbroadcast(dc, a.shape),  db = unbroadcast(dc, b.shape)
void add_broad_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) {
        TensorEngine *da = unbroadcast_to(grad_output, a->shape, a->ndim);
        accumulate_grad(a, da);
        free_tensor(da);
    }
    if (b) {
        TensorEngine *db = unbroadcast_to(grad_output, b->shape, b->ndim);
        accumulate_grad(b, db);
        free_tensor(db);
    }
}

// c = a - b (broadcast)  =>  da = unbroadcast(dc, a.shape),  db = unbroadcast(-dc, b.shape)
void sub_broad_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) {
        TensorEngine *da = unbroadcast_to(grad_output, a->shape, a->ndim);
        accumulate_grad(a, da);
        free_tensor(da);
    }
    if (b) {
        TensorEngine *neg_grad = negate_tensor(grad_output);
        TensorEngine *db = unbroadcast_to(neg_grad, b->shape, b->ndim);
        accumulate_grad(b, db);
        free_tensor(neg_grad);
        free_tensor(db);
    }
}

// c = a * b (broadcast)  =>  da = unbroadcast(dc * b, a.shape),  db = unbroadcast(dc * a, b.shape)
void mul_broad_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) {
        TensorEngine *grad_times_b = mlt_broad(grad_output, b);
        TensorEngine *da = unbroadcast_to(grad_times_b, a->shape, a->ndim);
        accumulate_grad(a, da);
        free_tensor(grad_times_b);
        free_tensor(da);
    }
    if (b) {
        TensorEngine *grad_times_a = mlt_broad(grad_output, a);
        TensorEngine *db = unbroadcast_to(grad_times_a, b->shape, b->ndim);
        accumulate_grad(b, db);
        free_tensor(grad_times_a);
        free_tensor(db);
    }
}

// c = a / b (broadcast)  =>  da = unbroadcast(dc / b, a.shape),  db = unbroadcast(-dc * a / b^2, b.shape)
void div_broad_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];

    if (a) {
        TensorEngine *grad_div_b = div_broad(grad_output, b);
        TensorEngine *da = unbroadcast_to(grad_div_b, a->shape, a->ndim);
        accumulate_grad(a, da);
        free_tensor(grad_div_b);
        free_tensor(da);
    }
    if (b) {
        TensorEngine *b_sq       = mlt_broad(b, b);
        TensorEngine *a_over_bsq = div_broad(a, b_sq);
        TensorEngine *dc_times   = mlt_broad(grad_output, a_over_bsq);
        TensorEngine *neg        = negate_tensor(dc_times);
        TensorEngine *db         = unbroadcast_to(neg, b->shape, b->ndim);

        accumulate_grad(b, db);
        free_tensor(b_sq);
        free_tensor(a_over_bsq);
        free_tensor(dc_times);
        free_tensor(neg);
        free_tensor(db);
    }
}

// ============================================================================
// Backward Implementations: Matrix Multiplication & Dot Product
// ============================================================================

void dot_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *a = self->saved_tensors[0];
    TensorEngine *b = self->saved_tensors[1];
    if (!a || !b) return;

    // 1D . 1D (Scalar inner product)
    if (a->ndim == 1 && b->ndim == 1) {
        TensorEngine *da     = mlt_broad(grad_output, b);
        TensorEngine *da_unb = unbroadcast_to(da, a->shape, a->ndim);
        accumulate_grad(a, da_unb);
        free_tensor(da);
        free_tensor(da_unb);

        TensorEngine *db     = mlt_broad(grad_output, a);
        TensorEngine *db_unb = unbroadcast_to(db, b->shape, b->ndim);
        accumulate_grad(b, db_unb);
        free_tensor(db);
        free_tensor(db_unb);
    }
    // 2D @ 1D (Matrix-Vector product)
    else if (a->ndim == 2 && b->ndim == 1) {
        // dA = grad_output.reshape([M, 1]) @ b.reshape([1, K])
        int sh_dy[] = { (int) a->shape[0], 1, N };
        TensorEngine *dy_col = reshape(grad_output, sh_dy, 2);

        int sh_b_row[] = { 1, (int) b->shape[0], N };
        TensorEngine *b_row = reshape(b, sh_b_row, 2);

        TensorEngine *da = dot_prod(dy_col, b_row);
        accumulate_grad(a, da);
        free_tensor(dy_col);
        free_tensor(b_row);
        free_tensor(da);

        // dB = A^T [K, M] @ grad_output [M] -> [K]
        TensorEngine *at = T(a);
        TensorEngine *db = dot_prod(at, grad_output);
        accumulate_grad(b, db);
        free_tensor(at);
        free_tensor(db);
    }
    // 1D @ 2D (Vector-Matrix product)
    else if (a->ndim == 1 && b->ndim == 2) {
        // da = grad_output [P] @ B^T [P, K] -> [K]
        TensorEngine *bt = T(b);
        TensorEngine *da = dot_prod(grad_output, bt);
        accumulate_grad(a, da);
        free_tensor(bt);
        free_tensor(da);

        // dB = a.reshape([K, 1]) @ grad_output.reshape([1, P]) -> [K, P]
        int sh_a_col[] = { (int) a->shape[0], 1, N };
        TensorEngine *a_col = reshape(a, sh_a_col, 2);

        int sh_dy_row[] = { 1, (int) b->shape[1], N };
        TensorEngine *dy_row = reshape(grad_output, sh_dy_row, 2);

        TensorEngine *db = dot_prod(a_col, dy_row);
        accumulate_grad(b, db);
        free_tensor(a_col);
        free_tensor(dy_row);
        free_tensor(db);
    }
    // 2D @ 2D / ND @ ND (Standard Matrix product: dA = dC @ B^T, dB = A^T @ dC)
    else {
        TensorEngine *bt = T(b);
        TensorEngine *da = dot_prod(grad_output, bt);
        accumulate_grad(a, da);
        free_tensor(bt);
        free_tensor(da);

        TensorEngine *at = T(a);
        TensorEngine *db = dot_prod(at, grad_output);
        accumulate_grad(b, db);
        free_tensor(at);
        free_tensor(db);
    }
}

// ============================================================================
// Backward Implementations: Structural / Tensor Layout Operations
// ============================================================================

// ---------- Transpose Backward ----------
typedef struct {
    int *axes;
    size_t ndim;
} TransposeCtx;

static void release_transpose_ctx(void *ctx) {
    TransposeCtx *tc = (TransposeCtx*) ctx;
    if (tc) {
        free(tc->axes);
        free(tc);
    }
}

void transpose_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;

    // 1D tensor was transposed to [N, 1]; reshape grad back to [N]
    if (input->ndim == 1) {
        TensorEngine *da = reshape(grad_output, input->shape, 1);
        accumulate_grad(input, da);
        free_tensor(da);
        return;
    }

    TransposeCtx *ctx = (TransposeCtx*) self->context;

    if (ctx == nullptr || ctx->axes == nullptr) {
        // Default transpose: reverse axes
        TensorEngine *da = T(grad_output);
        accumulate_grad(input, da);
        free_tensor(da);
    } else {
        // Invert the axes permutation
        int *inv_axes = (int*) malloc(sizeof(int) * ctx->ndim);
        for (size_t i = 0; i < ctx->ndim; i++) {
            inv_axes[ctx->axes[i]] = (int) i;
        }

        TensorEngine *da = T(grad_output, inv_axes);
        accumulate_grad(input, da);
        free(inv_axes);
        free_tensor(da);
    }
}

// ---------- Reshape Backward ----------
typedef struct {
    int *orig_shape;
    size_t orig_ndim;
} ReshapeCtx;

static void release_reshape_ctx(void *ctx) {
    ReshapeCtx *rc = (ReshapeCtx*) ctx;
    if (rc) {
        free(rc->orig_shape);
        free(rc);
    }
}

void reshape_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;

    ReshapeCtx *ctx = (ReshapeCtx*) self->context;
    TensorEngine *da = reshape(grad_output, ctx->orig_shape, ctx->orig_ndim);
    accumulate_grad(input, da);
    free_tensor(da);
}

// ---------- Slice Backward ----------
typedef struct {
    int *orig_shape;
    size_t orig_ndim;
    int *starts;
    int *steps;
    int *out_shape;
    int offset;
} SliceCtx;

static void release_slice_ctx(void *ctx) {
    SliceCtx *sc = (SliceCtx*) ctx;
    if (sc) {
        free(sc->orig_shape);
        free(sc->starts);
        free(sc->steps);
        free(sc->out_shape);
        free(sc);
    }
}

void slice_backward_fn(GradFn *self, TensorEngine *grad_output) {
    TensorEngine *input = self->saved_tensors[0];
    if (!input) return;

    SliceCtx *ctx = (SliceCtx*) self->context;

    // Create zero tensor matching the original unsliced shape
    int *sh = (int*) malloc(sizeof(int) * (ctx->orig_ndim + 1));
    for (size_t i = 0; i < ctx->orig_ndim; i++) {
        sh[i] = ctx->orig_shape[i];
    }
    sh[ctx->orig_ndim] = N;

    TensorEngine *da = zeros(sh, grad_output->__GPU__);
    free(sh);
    if (!da) return;

    // Scatter the sliced gradient back into the original tensor coordinates
    size_t out_size = grad_output->size;
    for (size_t linear = 0; linear < out_size; linear++) {
        size_t rem = linear;
        size_t in_offset = (size_t) ctx->offset;

        for (size_t d = ctx->orig_ndim; d-- > 0; ) {
            size_t coord = rem % (size_t) ctx->out_shape[d];
            rem /= (size_t) ctx->out_shape[d];
            int in_coord = ctx->starts[d] + (int) coord * ctx->steps[d];
            in_offset += (size_t) in_coord * (size_t) da->strides[d];
        }

        da->tensor[in_offset] += grad_output->tensor[linear];
    }

    accumulate_grad(input, da);
    free_tensor(da);
}

// ============================================================================
// Graph Attachment Helpers
// ============================================================================

void attach_binary_grad_fn(
    TensorEngine *result,
    TensorEngine *t1,
    TensorEngine *t2,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, TensorEngine *grad_output))
{
    if (!result || !t1 || !t2) return;

    result->requires_grad = true;
    result->is_leaf       = false;

    TensorEngine *saved[2] = { t1, t2 };
    result->grad_fn = create_grad_fn(name, type, apply, 2, saved, nullptr, nullptr);
    if (result->grad_fn) {
        result->gradfn = *result->grad_fn;
    }
}

void attach_unary_grad_fn(
    TensorEngine *result,
    TensorEngine *input,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, TensorEngine *grad_output),
    void *context,
    void (*release_context)(void *ctx))
{
    if (!result || !input) return;

    result->requires_grad = true;
    result->is_leaf       = false;

    TensorEngine *saved[1] = { input };
    result->grad_fn = create_grad_fn(name, type, apply, 1, saved, context, release_context);
    if (result->grad_fn) {
        result->gradfn = *result->grad_fn;
    }
}

void attach_transpose_grad_fn(TensorEngine *result, TensorEngine *input, const int *axes) {
    if (!result || !input) return;

    TransposeCtx *ctx = (TransposeCtx*) malloc(sizeof(TransposeCtx));
    if (ctx) {
        ctx->ndim = input->ndim;
        if (axes) {
            ctx->axes = (int*) malloc(sizeof(int) * input->ndim);
            for (size_t i = 0; i < input->ndim; i++) {
                ctx->axes[i] = axes[i];
            }
        } else {
            ctx->axes = nullptr;
        }
    }

    attach_unary_grad_fn(
        result, input, "TransposeBackward", OP_TRANSPOSE,
        transpose_backward_fn, ctx, release_transpose_ctx
    );
}

void attach_reshape_grad_fn(TensorEngine *result, TensorEngine *input, const int *orig_shape, size_t orig_ndim) {
    if (!result || !input) return;

    ReshapeCtx *ctx = (ReshapeCtx*) malloc(sizeof(ReshapeCtx));
    if (ctx) {
        ctx->orig_ndim  = orig_ndim;
        ctx->orig_shape = (int*) malloc(sizeof(int) * orig_ndim);
        for (size_t i = 0; i < orig_ndim; i++) {
            ctx->orig_shape[i] = orig_shape[i];
        }
    }

    attach_unary_grad_fn(
        result, input, "ReshapeBackward", OP_RESHAPE,
        reshape_backward_fn, ctx, release_reshape_ctx
    );
}

void attach_slice_grad_fn(
    TensorEngine *result,
    TensorEngine *input,
    const int *starts,
    const int *steps,
    const int *out_shape,
    int offset)
{
    if (!result || !input) return;

    SliceCtx *ctx = (SliceCtx*) malloc(sizeof(SliceCtx));
    if (ctx) {
        ctx->orig_ndim  = input->ndim;
        ctx->offset     = offset;
        ctx->orig_shape = (int*) malloc(sizeof(int) * input->ndim);
        ctx->starts     = (int*) malloc(sizeof(int) * input->ndim);
        ctx->steps      = (int*) malloc(sizeof(int) * input->ndim);
        ctx->out_shape  = (int*) malloc(sizeof(int) * input->ndim);

        for (size_t i = 0; i < input->ndim; i++) {
            ctx->orig_shape[i] = input->shape[i];
            ctx->starts[i]     = starts[i];
            ctx->steps[i]      = steps[i];
            ctx->out_shape[i]  = out_shape[i];
        }
    }

    attach_unary_grad_fn(
        result, input, "SliceBackward", OP_SLICE,
        slice_backward_fn, ctx, release_slice_ctx
    );
}

void attach_clamp_grad_fn(
    TensorEngine *result,
    TensorEngine *input,
    f64 low,
    f64 high)
{
    if (!result || !input) return;

    ClampCtx *ctx = (ClampCtx*) malloc(sizeof(ClampCtx));
    if (ctx) {
        ctx->low  = low;
        ctx->high = high;
    }

    attach_unary_grad_fn(
        result, input, "ClampBackward", OP_CLAMP,
        _clamp_backward_fn, ctx, release_clamp_ctx
    );
}
