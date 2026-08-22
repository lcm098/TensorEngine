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
