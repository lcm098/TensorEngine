# Tensor Engine

## backward
```
The `backward` function computes the gradients of a computational graph with 
respect to its leaf tensors using automatic differentiation (autograd). It performs 
a depth-first search (DFS) topological sort over the directed acyclic graph (DAG) 
rooted at `root`, and evaluates node backward functions in reverse topological 
order. The initial seed gradient `grad_output` defaults to a tensor of ones with the 
same shape and device target as `root` (when passed as `nullptr`). Computed 
gradients are accumulated into the `grad` field (`target->grad`) of each operand 
tensor in the graph. If `root` is `NULL`, an error message is printed to `stderr` 
and the function returns safely.
```

> Signature
```
void backward(TensorEngine *root, TensorEngine *grad_output = nullptr);
```

> Example
```
f64 da[] = {2.0, 3.0, E};
f64 db[] = {4.0, 5.0, E};
f64 dc[] = {1.0, 1.0, E};
int shape[] = {2, N};

TensorEngine* a = tensor(da, shape, false);
TensorEngine* b = tensor(db, shape, false);
TensorEngine* c = tensor(dc, shape, false);

TensorEngine* ab = mlt(a, b);
TensorEngine* z = add(ab, c);

backward(z);

p(a); // a->grad is [4.000000, 5.000000]
p(b); // b->grad is [2.000000, 3.000000]
p(c); // c->grad is [1.000000, 1.000000]

free_tensor(a);
free_tensor(b);
free_tensor(c);
free_tensor(ab);
free_tensor(z);
```

> Second Example
```
// Branching computation graph: z = (x + y) * (x - y) = x^2 - y^2
TensorEngine* x = tensor((f64[]){3.0, E}, (int[]){1, N}, false);
TensorEngine* y = tensor((f64[]){2.0, E}, (int[]){1, N}, false);

TensorEngine* sum = add(x, y);
TensorEngine* diff = sub(x, y);
TensorEngine* z = mlt(sum, diff);

backward(z);

p(x); // x->grad is [6.000000]
p(y); // y->grad is [-4.000000]

free_tensor(x);
free_tensor(y);
free_tensor(sum);
free_tensor(diff);
free_tensor(z);
```

## print_graph
```
The `print_graph` function recursively inspects and displays the computational 
graph DAG rooted at the given tensor `t` to standard output. It prints a tree-style 
hierarchy illustrating each operation node (such as `AddBackward`, `MulBackward`, 
`DotBackward`, `TransposeBackward`), tensor shapes, and leaf status (`[Leaf]` or 
`[Leaf: requires_grad=true]`), as well as whether a gradient (`has_grad`) has already 
been accumulated. A visited set prevents duplicate tree traversal in branching 
graphs. If `t` is `NULL`, an error message is output to `stderr`.
```

> Signature
```
void print_graph(TensorEngine *t);
```

> Example
```
TensorEngine* a = tensor((f64[]){1.0, 2.0, E}, (int[]){2, N}, false);
TensorEngine* b = tensor((f64[]){3.0, 4.0, E}, (int[]){2, N}, false);

TensorEngine* c = mlt(a, b);
TensorEngine* d = add(c, a);

print_graph(d);

free_tensor(a);
free_tensor(b);
free_tensor(c);
free_tensor(d);
```

> Second Example
```
TensorEngine* A = tensor((f64[]){1.0, 2.0, 3.0, 4.0, E}, (int[]){2, 2, N}, false);
TensorEngine* B = tensor((f64[]){5.0, 6.0, 7.0, 8.0, E}, (int[]){2, 2, N}, false);

TensorEngine* C = dot_prod(A, B);
print_graph(C);

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

## set_requires_grad
```
The `set_requires_grad` function sets the `requires_grad` boolean flag on a 
`TensorEngine` instance `t` and marks it as a leaf node (`is_leaf = true`). When 
set to `true`, the tensor participates as a tracked parameter in computational 
graphs for automatic differentiation. When set to `false`, the tensor can be 
excluded from gradient updates. If `t` is `NULL`, the function returns without 
performing any action.
```

> Signature
```
void set_requires_grad(TensorEngine *t, bool requires_grad);
```

> Example
```
TensorEngine* t = tensor((f64[]){1.0, 2.0, 3.0, E}, (int[]){3, N}, false);
set_requires_grad(t, true);

p(t); // displays 'requires_grad: true'

free_tensor(t);
```

> Second Example
```
TensorEngine* weights = zeros((int[]){10, 5, N}, false);
set_requires_grad(weights, true);

TensorEngine* bias = zeros((int[]){5, N}, false);
set_requires_grad(bias, false);

free_tensor(weights);
free_tensor(bias);
```

## zero_grad
```
The `zero_grad` function resets and frees any gradient tensor currently stored in 
the `grad` field (`t->grad`) of a `TensorEngine` instance `t`. It recursively calls 
`free_tensor` on `t->grad` to reclaim heap-allocated memory and sets `t->grad` to 
`nullptr`. This function is typically called between optimization steps or iterations 
to clear accumulated gradients before executing a new backward pass. If `t` is 
`NULL` or `t->grad` is already `nullptr`, the function returns safely.
```

> Signature
```
void zero_grad(TensorEngine *t);
```

> Example
```
TensorEngine* x = tensor((f64[]){3.0, 4.0, E}, (int[]){2, N}, false);
TensorEngine* y = mlt(x, x);

backward(y);
p(x); // x->grad is [6.000000, 8.000000]

zero_grad(x);
p(x); // x->grad is cleared (nullptr)

free_tensor(y);
free_tensor(x);
```

> Second Example
```
TensorEngine* w = ones((int[]){2, 2, N}, false);
TensorEngine* out = mlt(w, w);
backward(out);

// Reset gradient before next forward-backward cycle
zero_grad(w);
free_tensor(out);

TensorEngine* out2 = add(w, w);
backward(out2);
p(w); // w->grad is [1.000000, 1.000000, 1.000000, 1.000000]

free_tensor(out2);
free_tensor(w);
```

## create_grad_fn
```
The `create_grad_fn` function allocates and initializes a new `GradFn` node structure 
for the dynamic computational graph. It records the operation name string `name`, 
the operation type identifier enum `type`, the backward callback function pointer 
`apply`, an array of saved input tensor pointers `saved_tensors` of length `num_saved`, 
an optional operation-specific context pointer `context`, and an optional cleanup 
function `release_context`. Returns a newly allocated `GradFn` pointer, or `NULL` if 
memory allocation fails.
```

> Signature
```
GradFn* create_grad_fn(
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, TensorEngine *grad_output),
    size_t num_saved,
    TensorEngine **saved_tensors,
    void *context,
    void (*release_context)(void *context)
);
```

> Example
```
TensorEngine* a = tensor((f64[]){1.0, E}, (int[]){1, N}, false);
TensorEngine* b = tensor((f64[]){2.0, E}, (int[]){1, N}, false);
TensorEngine* saved[] = {a, b};

GradFn* fn = create_grad_fn("AddBackward", OP_ADD, add_backward_fn, 2, saved, nullptr, nullptr);

free_grad_fn(fn);
free_tensor(a);
free_tensor(b);
```

> Second Example
```
TensorEngine* x = tensor((f64[]){1.0, 2.0, E}, (int[]){2, N}, false);
TensorEngine* saved[] = {x};

GradFn* fn = create_grad_fn("CustomBackward", OP_CUSTOM, nullptr, 1, saved, nullptr, nullptr);

free_grad_fn(fn);
free_tensor(x);
```

## free_grad_fn
```
The `free_grad_fn` function deallocates heap memory associated with a `GradFn` node. 
If a `release_context` callback and non-null `context` pointer are present, the 
callback is invoked first to release any auxiliary context structures (such as 
transposition axes or slice bounds). It then frees the `saved_tensors` pointer array 
and the `GradFn` struct itself. Note that it does not free the saved tensors 
themselves, as tensor memory is managed by `free_tensor`. If `fn` is `NULL`, the 
function returns safely.
```

> Signature
```
void free_grad_fn(GradFn *fn);
```

> Example
```
TensorEngine* a = ones((int[]){2, 2, N}, false);
TensorEngine* saved[] = {a};
GradFn* fn = create_grad_fn("CustomOp", OP_CUSTOM, nullptr, 1, saved, nullptr, nullptr);

free_grad_fn(fn);
free_tensor(a);
```

> Second Example
```
GradFn* fn = nullptr;
free_grad_fn(fn); // Safe no-op
```

## accumulate_grad
```
The `accumulate_grad` function accumulates an incoming gradient tensor `incoming` into 
the `grad` field of a target tensor `target` (`target->grad += incoming`). If 
`target->grad` is currently `nullptr`, a clone of `incoming` is created and assigned 
directly to `target->grad`. If `target->grad` already exists, the function performs 
element-wise addition (using `add` when shapes match, or `add_broad` when shapes 
require broadcast alignment) and updates `target->grad` with the new sum. If either 
`target` or `incoming` is `NULL`, the function returns without performing any action.
```

> Signature
```
void accumulate_grad(TensorEngine *target, TensorEngine *incoming);
```

> Example
```
TensorEngine* target = tensor((f64[]){1.0, 2.0, E}, (int[]){2, N}, false);
TensorEngine* g1 = tensor((f64[]){0.5, 0.5, E}, (int[]){2, N}, false);
TensorEngine* g2 = tensor((f64[]){1.0, 1.0, E}, (int[]){2, N}, false);

accumulate_grad(target, g1); // target->grad becomes [0.5, 0.5]
accumulate_grad(target, g2); // target->grad becomes [1.5, 1.5]

p(target);

free_tensor(g1);
free_tensor(g2);
free_tensor(target);
```

> Second Example
```
TensorEngine* target = ones((int[]){2, 3, N}, false);
TensorEngine* inc = full((int[]){2, 3, N}, 2.0, false);

accumulate_grad(target, inc);
p(target->grad); // target->grad is filled with 2.0

free_tensor(inc);
free_tensor(target);
```

## unbroadcast_to
```
The `unbroadcast_to` function reduces an incoming gradient tensor `grad` across 
broadcasted dimensions by summation, matching the target tensor shape defined by 
`target_shape` and `target_ndim`. When a forward operation broadcasts smaller or 
lower-dimensional tensors to a larger common shape, the backward pass must sum the 
gradients along all expanded singleton dimensions (size 1) and prepended leading 
dimensions to preserve shape consistency with the original operand. If `grad` 
already has the exact target shape, a cloned copy is returned. Returns a newly 
allocated `TensorEngine` pointer with the target shape, or `NULL` if allocation fails.
```

> Signature
```
TensorEngine* unbroadcast_to(TensorEngine *grad, const int *target_shape, size_t target_ndim);
```

> Example
```
// grad has shape [2, 3], target shape is [1, 3]
f64 data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int grad_shape[] = {2, 3, N};
TensorEngine* grad = tensor(data, grad_shape, false);

int target_shape[] = {1, 3};
TensorEngine* unb = unbroadcast_to(grad, target_shape, 2);

p(unb); // Shape [1, 3], values [5.000000, 7.000000, 9.000000]

free_tensor(grad);
free_tensor(unb);
```

> Second Example
```
// grad has shape [2, 3], target shape is 1D [3]
f64 data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int grad_shape[] = {2, 3, N};
TensorEngine* grad = tensor(data, grad_shape, false);

int target_shape[] = {3};
TensorEngine* unb = unbroadcast_to(grad, target_shape, 1);

p(unb); // Shape [3], values [5.000000, 7.000000, 9.000000]

free_tensor(grad);
free_tensor(unb);
```

## attach_binary_grad_fn
```
The `attach_binary_grad_fn` helper attaches a two-input binary backward node (`GradFn`) 
to a result tensor resulting from operations like `add`, `sub`, `mlt`, `divt`, `dot_prod`, 
or broadcast variants. It saves references to the two input operands `t1` and `t2`, sets 
`result->requires_grad = true`, sets `result->is_leaf = false`, and associates the 
specified backward callback function pointer `apply`.
```

> Signature
```
void attach_binary_grad_fn(
    TensorEngine *result,
    TensorEngine *t1,
    TensorEngine *t2,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, TensorEngine *grad_output)
);
```

> Example
```
TensorEngine* t1 = tensor((f64[]){1.0, E}, (int[]){1, N}, false);
TensorEngine* t2 = tensor((f64[]){2.0, E}, (int[]){1, N}, false);
TensorEngine* res = tensor((f64[]){3.0, E}, (int[]){1, N}, false);

attach_binary_grad_fn(res, t1, t2, "AddBackward", OP_ADD, add_backward_fn);

print_graph(res);

free_tensor(t1);
free_tensor(t2);
free_tensor(res);
```

> Second Example
```
TensorEngine* a = ones((int[]){2, 2, N}, false);
TensorEngine* b = ones((int[]){2, 2, N}, false);
TensorEngine* c = ones((int[]){2, 2, N}, false);

attach_binary_grad_fn(c, a, b, "MulBackward", OP_MUL, mul_backward_fn);

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

## attach_unary_grad_fn
```
The `attach_unary_grad_fn` helper attaches a single-input backward node (`GradFn`) 
to a result tensor resulting from unary operations (such as `transpose`, `reshape`, 
`slice`). It saves a reference to `input`, sets `result->requires_grad = true` and 
`result->is_leaf = false`, and attaches custom operation context along with its 
cleanup callback `release_context`.
```

> Signature
```
void attach_unary_grad_fn(
    TensorEngine *result,
    TensorEngine *input,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, TensorEngine *grad_output),
    void *context,
    void (*release_context)(void *ctx)
);
```

> Example
```
TensorEngine* input = ones((int[]){2, 3, N}, false);
TensorEngine* res = ones((int[]){3, 2, N}, false);

attach_unary_grad_fn(res, input, "TransposeBackward", OP_TRANSPOSE, transpose_backward_fn, nullptr, nullptr);

free_tensor(input);
free_tensor(res);
```

> Second Example
```
TensorEngine* input = ones((int[]){6, N}, false);
TensorEngine* res = ones((int[]){2, 3, N}, false);

attach_unary_grad_fn(res, input, "ReshapeBackward", OP_RESHAPE, reshape_backward_fn, nullptr, nullptr);

free_tensor(input);
free_tensor(res);
```

## attach_transpose_grad_fn
```
The `attach_transpose_grad_fn` helper constructs and attaches a `TransposeBackward` 
graph node to a transposed result tensor. It packages the original tensor's dimension 
count and the permutation axes array into a dedicated context struct so that during 
the backward pass, the gradient can be inverted back to the input tensor's layout.
```

> Signature
```
void attach_transpose_grad_fn(TensorEngine *result, TensorEngine *input, const int *axes);
```

> Example
```
TensorEngine* t = tensor((f64[]){1.0, 2.0, 3.0, 4.0, E}, (int[]){2, 2, N}, false);
TensorEngine* res = T(t);

// attach_transpose_grad_fn is automatically invoked inside T() / transpose()
p(res);

free_tensor(t);
free_tensor(res);
```

> Second Example
```
TensorEngine* t = arange(1.0, 25.0, 1.0, false);
int sh[] = {2, 3, 4};
TensorEngine* t3d = reshape(t, sh, 3);
int axes[] = {2, 0, 1};

TensorEngine* transposed = T(t3d, axes);

free_tensor(t);
free_tensor(t3d);
free_tensor(transposed);
```

## attach_reshape_grad_fn
```
The `attach_reshape_grad_fn` helper packages the original shape and dimensionality 
of `input` into a reshape context and attaches a `ReshapeBackward` node to `result`. 
During `backward`, the incoming gradient is reshaped back to match `input`'s original shape.
```

> Signature
```
void attach_reshape_grad_fn(TensorEngine *result, TensorEngine *input, const int *orig_shape, size_t orig_ndim);
```

> Example
```
TensorEngine* t = arange(1.0, 7.0, 1.0, false);
int sh[] = {2, 3};
TensorEngine* reshaped = reshape(t, sh, 2);

print_graph(reshaped);

free_tensor(t);
free_tensor(reshaped);
```

> Second Example
```
TensorEngine* t = ones((int[]){2, 2, 2, N}, false);
int sh[] = {8};
TensorEngine* flat = reshape(t, sh, 1);

backward(flat);
p(t); // t->grad is restored in [2, 2, 2] shape

free_tensor(t);
free_tensor(flat);
```

## attach_slice_grad_fn
```
The `attach_slice_grad_fn` helper packages slicing coordinates (`starts`, `steps`, 
`out_shape`, `offset`) and original tensor shape into a slice context, and attaches 
a `SliceBackward` node to `result`. During backpropagation, the sliced gradient values 
are scattered back into their exact coordinate positions in a zero tensor matching `input`.
```

> Signature
```
void attach_slice_grad_fn(
    TensorEngine *result,
    TensorEngine *input,
    const int *starts,
    const int *steps,
    const int *out_shape,
    int offset
);
```

> Example
```
TensorEngine* x = tensor((f64[]){10.0, 20.0, 30.0, 40.0, E}, (int[]){4, N}, false);
int s[][3] = { {1, 3, 1} };
TensorEngine* sl = slice(x, s, 1, 0); // slice elements at index 1 and 2
backward(sl);

p(x); // x->grad is [0.0, 1.0, 1.0, 0.0]

free_tensor(x);
free_tensor(sl);
```

> Second Example
```
TensorEngine* x = ones((int[]){4, 4, N}, false);
int s[][3] = { {0, 2, 1}, {0, 2, 1} };
TensorEngine* sl = slice(x, s, 2, 0);
backward(sl);

p(x); // x->grad has 1.0 on top-left 2x2 block and 0.0 elsewhere

free_tensor(x);
free_tensor(sl);
```

## add_backward_fn
```
The `add_backward_fn` is the backward callback function for element-wise addition 
(c = a + b). By the sum rule of differential calculus:
    ∂c/∂a = 1.0  =>  grad_a = grad_output
    ∂c/∂b = 1.0  =>  grad_b = grad_output
The incoming gradient `grad_output` is accumulated directly into the gradient 
buffers of both input operands `a` and `b`.
```

> Signature
```
void add_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* a = tensor((f64[]){2.0, E}, (int[]){1, N}, false);
TensorEngine* b = tensor((f64[]){3.0, E}, (int[]){1, N}, false);
TensorEngine* c = add(a, b);
backward(c);

p(a); // a->grad is [1.000000]
p(b); // b->grad is [1.000000]

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

> Second Example
```
TensorEngine* a = ones((int[]){2, 2, N}, false);
TensorEngine* b = full((int[]){2, 2, N}, 5.0, false);
TensorEngine* c = add(a, b);
backward(c);

p(a); // a->grad is [[1, 1], [1, 1]]

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

## sub_backward_fn
```
The `sub_backward_fn` is the backward callback function for element-wise subtraction 
(c = a - b). By the difference rule of calculus:
    ∂c/∂a =  1.0  =>  grad_a =  grad_output
    ∂c/∂b = -1.0  =>  grad_b = -grad_output
It accumulates `grad_output` into `a->grad` and `-grad_output` into `b->grad`.
```

> Signature
```
void sub_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* a = tensor((f64[]){10.0, E}, (int[]){1, N}, false);
TensorEngine* b = tensor((f64[]){4.0, E}, (int[]){1, N}, false);
TensorEngine* c = sub(a, b);
backward(c);

p(a); // a->grad is [1.000000]
p(b); // b->grad is [-1.000000]

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

> Second Example
```
TensorEngine* a = ones((int[]){3, N}, false);
TensorEngine* b = ones((int[]){3, N}, false);
TensorEngine* c = sub(a, b);
backward(c);

p(b); // b->grad is [-1.000000, -1.000000, -1.000000]

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

## mul_backward_fn
```
The `mul_backward_fn` is the backward callback function for element-wise multiplication 
(c = a * b). By the product rule of calculus:
    ∂c/∂a = b  =>  grad_a = grad_output * b
    ∂c/∂b = a  =>  grad_b = grad_output * a
It computes `grad_output * b` and accumulates it into `a->grad`, and computes 
`grad_output * a` and accumulates it into `b->grad`.
```

> Signature
```
void mul_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* a = tensor((f64[]){3.0, 4.0, E}, (int[]){2, N}, false);
TensorEngine* b = tensor((f64[]){5.0, 6.0, E}, (int[]){2, N}, false);
TensorEngine* c = mlt(a, b);
backward(c);

p(a); // a->grad is [5.000000, 6.000000]
p(b); // b->grad is [3.000000, 4.000000]

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

> Second Example
```
TensorEngine* x = tensor((f64[]){4.0, E}, (int[]){1, N}, false);
TensorEngine* y = mlt(x, x); // y = x^2
backward(y);

p(x); // x->grad is [8.000000] (2 * x)

free_tensor(x);
free_tensor(y);
```

## div_backward_fn
```
The `div_backward_fn` is the backward callback function for element-wise division 
(c = a / b). By the quotient rule of calculus:
    ∂c/∂a =  1 / b      =>  grad_a =  grad_output / b
    ∂c/∂b = -a / (b^2)  =>  grad_b = -grad_output * a / (b^2)
It computes `grad_output / b` for `a->grad`, and `-grad_output * a / (b^2)` 
for `b->grad`.
```

> Signature
```
void div_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* a = tensor((f64[]){12.0, E}, (int[]){1, N}, false);
TensorEngine* b = tensor((f64[]){4.0, E}, (int[]){1, N}, false);
TensorEngine* c = divt(a, b);
backward(c);

p(a); // a->grad is [0.250000] (1/4)
p(b); // b->grad is [-0.750000] (-12/16)

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

> Second Example
```
TensorEngine* a = tensor((f64[]){6.0, 8.0, E}, (int[]){2, N}, false);
TensorEngine* b = tensor((f64[]){2.0, 2.0, E}, (int[]){2, N}, false);
TensorEngine* c = divt(a, b);
backward(c);

p(a); // a->grad is [0.500000, 0.500000]
p(b); // b->grad is [-1.500000, -2.000000]

free_tensor(a);
free_tensor(b);
free_tensor(c);
```

## add_broad_backward_fn
```
The `add_broad_backward_fn` is the backward callback for broadcast addition 
(c = a + b where a and b have broadcast-compatible shapes):
    grad_a = unbroadcast_to(grad_output, a.shape)
    grad_b = unbroadcast_to(grad_output, b.shape)
Incoming gradients are sum-reduced along broadcasted singleton (size 1) and 
prepended dimensions to restore the original operand shapes.
```

> Signature
```
void add_broad_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* A = ones((int[]){2, 3, N}, false);
TensorEngine* B = ones((int[]){1, 3, N}, false);
TensorEngine* C = add_broad(A, B);
backward(C);

p(A); // A->grad is shape [2, 3] filled with 1.0
p(B); // B->grad is shape [1, 3] filled with 2.0 (summed along dim 0)

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

> Second Example
```
TensorEngine* A = full((int[]){3, 1, N}, 2.0, false);
TensorEngine* B = full((int[]){1, 4, N}, 3.0, false);
TensorEngine* C = add_broad(A, B);
backward(C);

p(A); // A->grad is [4.0, 4.0, 4.0] (summed along dim 1)
p(B); // B->grad is [3.0, 3.0, 3.0, 3.0] (summed along dim 0)

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

## sub_broad_backward_fn
```
The `sub_broad_backward_fn` is the backward callback for broadcast subtraction 
(c = a - b where a and b have broadcast-compatible shapes):
    grad_a = unbroadcast_to( grad_output, a.shape)
    grad_b = unbroadcast_to(-grad_output, b.shape)
It reduces `grad_output` to `a`'s shape for `a->grad`, and reduces `-grad_output` 
to `b`'s shape for `b->grad`.
```

> Signature
```
void sub_broad_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* A = ones((int[]){2, 3, N}, false);
TensorEngine* B = ones((int[]){1, 3, N}, false);
TensorEngine* C = sub_broad(A, B);
backward(C);

p(B); // B->grad is shape [1, 3] filled with -2.0

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

> Second Example
```
TensorEngine* A = full((int[]){2, 2, N}, 5.0, false);
TensorEngine* B = full((int[]){2, N}, 2.0, false);
TensorEngine* C = sub_broad(A, B);
backward(C);

p(B); // B->grad is [-2.0, -2.0]

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

## mul_broad_backward_fn
```
The `mul_broad_backward_fn` is the backward callback for broadcast multiplication 
(c = a * b where a and b have broadcast-compatible shapes):
    grad_a = unbroadcast_to(grad_output * b, a.shape)
    grad_b = unbroadcast_to(grad_output * a, b.shape)
It multiplies `grad_output` by the opposing operand and sum-reduces the broadcasted 
dimensions to match each operand's original shape.
```

> Signature
```
void mul_broad_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* A = tensor((f64[]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E}, (int[]){2, 3, N}, false);
TensorEngine* B = tensor((f64[]){2.0, 2.0, 2.0, E}, (int[]){1, 3, N}, false);
TensorEngine* C = mlt_broad(A, B);
backward(C);

p(B); // B->grad is shape [1, 3] with values [5.000000, 7.000000, 9.000000]

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

> Second Example
```
TensorEngine* A = full((int[]){3, 1, N}, 4.0, false);
TensorEngine* B = full((int[]){1, 2, N}, 5.0, false);
TensorEngine* C = mlt_broad(A, B);
backward(C);

p(A); // A->grad is [10.0, 10.0, 10.0] (5.0 * 2)

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

## div_broad_backward_fn
```
The `div_broad_backward_fn` is the backward callback for broadcast division 
(c = a / b where a and b have broadcast-compatible shapes):
    grad_a = unbroadcast_to( grad_output / b,          a.shape)
    grad_b = unbroadcast_to(-grad_output * a / (b^2),  b.shape)
It computes the quotient gradients and sum-reduces them along broadcasted dimensions.
```

> Signature
```
void div_broad_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* A = tensor((f64[]){10.0, 20.0, 30.0, 40.0, E}, (int[]){2, 2, N}, false);
TensorEngine* B = tensor((f64[]){2.0, 2.0, E}, (int[]){1, 2, N}, false);
TensorEngine* C = div_broad(A, B);
backward(C);

p(A); // A->grad is shape [2, 2] with values [0.5, 0.5, 0.5, 0.5]
p(B); // B->grad is shape [1, 2] with values [-10.0, -15.0]

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

> Second Example
```
TensorEngine* A = ones((int[]){3, 3, N}, false);
TensorEngine* B = full((int[]){1, 3, N}, 5.0, false);
TensorEngine* C = div_broad(A, B);
backward(C);

p(A); // A->grad is [0.2, 0.2, ...]

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

## dot_backward_fn
```
The `dot_backward_fn` is the backward callback for dot products and matrix 
multiplications:
- For 1D inner product (c = a · b, scalar [1]):
    grad_a = grad_output * b
    grad_b = grad_output * a
- For 2D matrix multiplication (C = A @ B, where A is [M, K] and B is [K, P]):
    ∂L/∂A = grad_output @ B^T   (shape [M, K])
    ∂L/∂B = A^T @ grad_output   (shape [K, P])
- For 2D @ 1D matrix-vector product (y = A @ x, where A is [M, K] and x is [K]):
    ∂L/∂A = grad_output.reshape([M, 1]) @ x.reshape([1, K])
    ∂L/∂x = A^T @ grad_output
- For 1D @ 2D vector-matrix product (y = x @ B, where x is [K] and B is [K, P]):
    ∂L/∂x = grad_output @ B^T
    ∂L/∂B = x.reshape([K, 1]) @ grad_output.reshape([1, P])
```

> Signature
```
void dot_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
f64 dA[] = {1, 2, 3, 4, E};
f64 dB[] = {5, 6, 7, 8, E};
int sh[] = {2, 2, N};

TensorEngine* A = tensor(dA, sh, false);
TensorEngine* B = tensor(dB, sh, false);
TensorEngine* C = dot_prod(A, B);

backward(C);

p(A); // A->grad is [[11.0, 15.0], [11.0, 15.0]]
p(B); // B->grad is [[4.0, 4.0], [6.0, 6.0]]

free_tensor(A);
free_tensor(B);
free_tensor(C);
```

> Second Example
```
// 2D @ 1D Matrix-Vector product
TensorEngine* A = tensor((f64[]){1, 2, 3, 4, E}, (int[]){2, 2, N}, false);
TensorEngine* x = tensor((f64[]){2, 3, E}, (int[]){2, N}, false);
TensorEngine* y = dot_prod(A, x);

backward(y);

p(A); // A->grad is [[2.0, 3.0], [2.0, 3.0]]
p(x); // x->grad is [4.0, 6.0]

free_tensor(A);
free_tensor(x);
free_tensor(y);
```

## transpose_backward_fn
```
The `transpose_backward_fn` is the backward callback for tensor transpositions 
(y = T(x, axes)):
    grad_x = T(grad_output, inv_axes)
It applies the inverse permutation (or reverses dimensions for default transpose) 
to `grad_output` and accumulates the result into `input->grad`.
```

> Signature
```
void transpose_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* x = tensor((f64[]){1, 2, 3, 4, 5, 6, E}, (int[]){2, 3, N}, false);
TensorEngine* y = T(x);
backward(y);

p(x); // x->grad has shape [2, 3] filled with 1.0

free_tensor(x);
free_tensor(y);
```

> Second Example
```
TensorEngine* x = tensor((f64[]){1, 2, 3, 4, E}, (int[]){4, N}, false);
TensorEngine* col = T(x); // Transposes 1D [4] to 2D [4, 1]
backward(col);

p(x); // x->grad is restored in 1D [4] shape

free_tensor(x);
free_tensor(col);
```

## reshape_backward_fn
```
The `reshape_backward_fn` is the backward callback for tensor reshaping 
(y = reshape(x, new_shape)):
    grad_x = reshape(grad_output, orig_shape)
It restores the incoming gradient tensor back into the input tensor's original 
multidimensional shape.
```

> Signature
```
void reshape_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* x = arange(1.0, 7.0, 1.0, false);
int sh[] = {2, 3};
TensorEngine* y = reshape(x, sh, 2);
backward(y);

p(x); // x->grad is in 1D [6] shape

free_tensor(x);
free_tensor(y);
```

> Second Example
```
TensorEngine* x = ones((int[]){2, 3, 4, N}, false);
int sh[] = {24};
TensorEngine* y = reshape(x, sh, 1);
backward(y);

p(x); // x->grad is in 3D [2, 3, 4] shape

free_tensor(x);
free_tensor(y);
```

## slice_backward_fn
```
The `slice_backward_fn` is the backward callback for tensor slicing 
(y = slice(x, indices)):
    grad_x[starts:stops:steps] += grad_output
It creates a zero tensor matching `x`'s original shape and scatters the sliced 
gradient values back into their exact original coordinate positions.
```

> Signature
```
void slice_backward_fn(GradFn *self, TensorEngine *grad_output);
```

> Example
```
TensorEngine* x = tensor((f64[]){10, 20, 30, 40, E}, (int[]){4, N}, false);
int s[][3] = { {1, 3, 1} };
TensorEngine* sl = slice(x, s, 1, 0); // slice elements at index 1 and 2
backward(sl);

p(x); // x->grad is [0.0, 1.0, 1.0, 0.0]

free_tensor(x);
free_tensor(sl);
```

> Second Example
```
TensorEngine* mat = ones((int[]){3, 3, N}, false);
int s[][3] = { {0, 2, 1}, {0, 2, 1} };
TensorEngine* sub = slice(mat, s, 2, 0);
backward(sub);

p(mat); // mat->grad has 1.0 on top-left 2x2 block and 0.0 elsewhere

free_tensor(mat);
free_tensor(sub);
```
