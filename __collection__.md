# Tensor Engine - API Collection Documentation

This document provides reference documentation for all non-static functions implemented across `linear.cpp`, `pipeline.cpp`, `initializer.cpp`, `math.cpp`, and `gradfn.cpp`.

---

# Table of Contents
- [Linear Layer Functions (`linear.cpp`)](#linear-layer-functions-linearcpp)
  - [Linear_Lay](#linear_lay)
  - [free_layer](#free_layer)
- [Sequential Pipeline Functions (`pipeline.cpp`)](#sequential-pipeline-functions-pipelinecpp)
  - [build](#build)
  - [print_pipeline](#print_pipeline)
  - [free_pipeline](#free_pipeline)
- [Tensor & Bias Initializer Functions (`initializer.cpp`)](#tensor--bias-initializer-functions-initializercpp)
  - [initialize_tensor](#initialize_tensor)
  - [initialize_bias](#initialize_bias)
- [Mathematical Functions (`math.cpp`)](#mathematical-functions-mathcpp)
  - [_cos](#_cos)
  - [_sin](#_sin)
  - [_clamp](#_clamp)
- [Autograd Engine & Graph Functions (`gradfn.cpp`)](#autograd-engine--graph-functions-gradfncpp)
  - [Autograd Lifecycle & Traversal](#autograd-lifecycle--traversal)
    - [set_requires_grad](#set_requires_grad)
    - [zero_grad](#zero_grad)
    - [backward](#backward)
    - [print_graph](#print_graph)
  - [Graph Node Construction & Helper Functions](#graph-node-construction--helper-functions)
    - [create_grad_fn](#create_grad_fn)
    - [free_grad_fn](#free_grad_fn)
    - [accumulate_grad](#accumulate_grad)
    - [unbroadcast_to](#unbroadcast_to)
  - [Graph Attachment Functions](#graph-attachment-functions)
    - [attach_binary_grad_fn](#attach_binary_grad_fn)
    - [attach_unary_grad_fn](#attach_unary_grad_fn)
    - [attach_transpose_grad_fn](#attach_transpose_grad_fn)
    - [attach_reshape_grad_fn](#attach_reshape_grad_fn)
    - [attach_slice_grad_fn](#attach_slice_grad_fn)
    - [attach_clamp_grad_fn](#attach_clamp_grad_fn)
  - [Backward Functions](#backward-functions)
    - [add_backward_fn](#add_backward_fn)
    - [sub_backward_fn](#sub_backward_fn)
    - [mul_backward_fn](#mul_backward_fn)
    - [div_backward_fn](#div_backward_fn)
    - [_cos_backward_fn](#_cos_backward_fn)
    - [_sin_backward_fn](#_sin_backward_fn)
    - [_clamp_backward_fn](#_clamp_backward_fn)
    - [add_broad_backward_fn](#add_broad_backward_fn)
    - [sub_broad_backward_fn](#sub_broad_backward_fn)
    - [mul_broad_backward_fn](#mul_broad_backward_fn)
    - [div_broad_backward_fn](#div_broad_backward_fn)
    - [dot_backward_fn](#dot_backward_fn)
    - [transpose_backward_fn](#transpose_backward_fn)
    - [reshape_backward_fn](#reshape_backward_fn)
    - [slice_backward_fn](#slice_backward_fn)

---

# Linear Layer Functions (`linear.cpp`)

## Linear_Lay
```
The `Linear_Lay` function allocates and configures a fully connected linear layer (`Layer`). 
It initializes the weight tensor with the specified shape `[in_feature, out_feature]` using 
the provided `InitType` weight initializer strategy, and initializes the bias tensor with 
shape `[1, out_feature]` using the specified `BiasInitType` and hyperparameters. The desired 
activation function type `ActivationType` is stored in the layer descriptor. Note that no matrix 
computation or activation transformation is performed at construction time; the function solely 
packages weights, bias, dimensions, and activation mode into a heap-allocated `Layer` object.
```

> Signature
```c
Layer* Linear_Lay(
    int in_feature,
    int out_feature,
    InitType BIY,
    ActivationType AT,
    BiasInitType BIT,
    f64 bias_param1,
    f64 bias_param2,
    bool __GPU__
);
```

> Example
```c
Layer *layer = Linear_Lay(
    784, 128,
    INIT_XAVIER_NORMAL,
    ReLU,
    BIAS_CONSTANT, 0.01, 0.0,
    false
);

p(layer->weights);
p(layer->bias);

free_layer(layer);
```

---

## free_layer
```
The `free_layer` function releases all heap-allocated resources associated with a `Layer` 
structure. It safely frees the underlying data arrays (handling device memory deallocation 
via `freeDeviceMemory` if allocated on the GPU, or host memory via `free`), shape arrays, 
strides arrays, the individual `TensorEngine` weight and bias structs, and finally the `Layer` 
struct itself. If a `NULL` pointer is passed, the function returns immediately without error.
```

> Signature
```c
void free_layer(Layer *layer);
```

> Example
```c
Layer *layer = Linear_Lay(4, 2, INIT_HE_NORMAL, Sigmoid, BIAS_ZEROS, 0.0, 0.0, false);
// ... use layer in forward / backward passes ...
free_layer(layer);
```

---

# Sequential Pipeline Functions (`pipeline.cpp`)

## build
```
The `build` function constructs a multi-layer sequential neural network pipeline (`__pipeLine__`). 
It accepts a variadic list of pointers to `Layer` configurations (typically initialized via `Linear_Lay`), 
terminated by a mandatory trailing `NULL` pointer sentinel. The function determines the total depth 
of the pipeline, dynamically allocates the layer pointer array, and preserves the execution sequence 
of layers. If `first` is `NULL`, an error message is printed to `stderr` and `NULL` is returned.
```

> Signature
```c
__pipeLine__* build(Layer *first, ...);
```

> Example
```c
__pipeLine__ *pipe = build(
    Linear_Lay(784, 128, INIT_XAVIER_NORMAL, ReLU, BIAS_CONSTANT, 0.01, 0.0, false),
    Linear_Lay(128, 64,  INIT_XAVIER_NORMAL, ReLU, BIAS_CONSTANT, 0.01, 0.0, false),
    Linear_Lay(64,  10,  INIT_XAVIER_NORMAL, Softmax, BIAS_ZEROS, 0.0, 0.0, false),
    NULL
);

print_pipeline(pipe);
free_pipeline(pipe);
```

---

## print_pipeline
```
The `print_pipeline` function formats and prints structural diagnostics for an entire sequential 
pipeline to standard output. For each layer in the network, it displays its index, input feature 
dimension (`in_feature`), output feature dimension (`out_feature`), activation function name 
(e.g., "Sigmoid", "ReLU", "Tanh", "Softmax", "LeakyReLU", "ELU", "GELU", "Swish", "Linear"), 
as well as the shapes, element sizes, and GPU target flags of both weight and bias tensors.
```

> Signature
```c
void print_pipeline(__pipeLine__ *pipe);
```

> Example
```c
print_pipeline(pipe);
```

---

## free_pipeline
```
The `free_pipeline` function deallocates an entire `__pipeLine__` sequence and all of its constituent 
layers. It iterates over each layer, invoking `free_layer` to release weights, biases, and metadata, 
and then frees the internal layer array and the `__pipeLine__` structure itself. Passing a `NULL` 
pointer safely returns without operation.
```

> Signature
```c
void free_pipeline(__pipeLine__ *p);
```

> Example
```c
free_pipeline(pipe);
```

---

# Tensor & Bias Initializer Functions (`initializer.cpp`)

## initialize_tensor
```
The `initialize_tensor` function allocates and populates a new `TensorEngine` instance using 
advanced weight initialization heuristics. It automatically calculates fan-in and fan-out 
based on the supplied `shape` array, ensures the global pseudo-random number generator is seeded, 
and generates random numbers according to the requested `InitType`:
- `INIT_XAVIER_UNIFORM` / `INIT_GLOROT_UNIFORM`: Uniform in [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
- `INIT_XAVIER_NORMAL` / `INIT_GLOROT_NORMAL`: Normal N(0, sqrt(2/(fan_in+fan_out)))
- `INIT_KAIMING_UNIFORM` / `INIT_HE_UNIFORM`: Uniform in [-sqrt(6/fan_in), sqrt(6/fan_in)]
- `INIT_KAIMING_NORMAL` / `INIT_HE_NORMAL`: Normal N(0, sqrt(2/fan_in))
- `INIT_LECUN_UNIFORM`: Uniform in [-sqrt(3/fan_in), sqrt(3/fan_in)]
- `INIT_LECUN_NORMAL`: Normal N(0, sqrt(1/fan_in))
Returns a pointer to the initialized `TensorEngine`, or `NULL` on validation/allocation failure.
```

> Signature
```c
TensorEngine* initialize_tensor(int shape[], InitType type, bool __GPU__);
```

> Example
```c
int shape[] = {64, 32, N};
TensorEngine *w = initialize_tensor(shape, INIT_KAIMING_NORMAL, false);
p(w);
free_tensor(w);
```

---

## initialize_bias
```
The `initialize_bias` function creates and initializes a bias tensor according to the specified 
`BiasInitType` distribution scheme:
- `BIAS_ZEROS`: Fills all elements with 0.0.
- `BIAS_ONES`: Fills all elements with 1.0.
- `BIAS_CONSTANT`: Fills all elements with `param1`.
- `BIAS_UNIFORM`: Fills elements with uniform random values in `[param1, param2]`.
- `BIAS_NORMAL`: Fills elements with normal random values from N(`param1`, `param2`).
- `BIAS_TRUNCATED_NORMAL`: Fills elements from normal distribution N(`param1`, `param2`) truncated within 2 standard deviations.
Returns a pointer to the newly allocated bias `TensorEngine`.
```

> Signature
```c
TensorEngine* initialize_bias(int shape[], BiasInitType type, f64 param1, f64 param2, bool __GPU__);
```

> Example
```c
int b_shape[] = {1, 64, N};
TensorEngine *bias = initialize_bias(b_shape, BIAS_NORMAL, 0.0, 0.1, false);
p(bias);
free_tensor(bias);
```

---

# Mathematical Functions (`math.cpp`)

## _cos
```
The `_cos` function computes the element-wise cosine of the input tensor `t1`. It employs full 
range reduction (modulo 2*pi and quadrant symmetry decomposition to [-pi/4, pi/4]) followed by 
a highly accurate Taylor polynomial approximation, maintaining strict bounds [-1.0, 1.0] across 
all real numbers without numerical explosion. It automatically constructs and attaches a unary 
computational graph node with name "CosBackward", type `OP_COS`, and backward callback 
`_cos_backward_fn` for autograd tracking.
```

> Signature
```c
TensorEngine* _cos(TensorEngine *t1);
```

> Example
```c
TensorEngine *x = tensor((f64[]){0.0, 1.0, 3.14159265, E}, (int[]){3, N}, false);
set_requires_grad(x, true);

TensorEngine *y = _cos(x);
p(y); // [1.000000, 0.540302, -1.000000]

backward(y);
p(x); // x->grad contains [-sin(x)] = [0.000000, -0.841471, 0.000000]

free_tensor(x);
free_tensor(y);
```

---

## _sin
```
The `_sin` function computes the element-wise sine of the input tensor `t1`. It employs full 
range reduction (modulo 2*pi followed by quadrant symmetry reduction to [-pi/4, pi/4]) and Taylor 
polynomial evaluation to ensure exact double-precision values bounded in [-1.0, 1.0]. It automatically 
attaches a unary computational graph node with name "SinBackward", operation enum `OP_SIN`, and 
backward callback `_sin_backward_fn`.
```

> Signature
```c
TensorEngine* _sin(TensorEngine *t1);
```

> Example
```c
TensorEngine *x = tensor((f64[]){0.0, 1.5707963, 3.14159265, E}, (int[]){3, N}, false);
set_requires_grad(x, true);

TensorEngine *y = _sin(x);
p(y); // [0.000000, 1.000000, 0.000000]

backward(y);
p(x); // x->grad contains [cos(x)] = [1.000000, 0.000000, -1.000000]

free_tensor(x);
free_tensor(y);
```

---

## _clamp
```
The `_clamp` function restricts all values in tensor `t1` to the numeric range between `Low` and 
`High`. Any element less than the lower bound is set to the lower bound, and any element greater 
than the upper bound is set to the upper bound. The function correctly resolves bounds regardless 
of parameter ordering. It attaches a unary computational graph node with name "ClampBackward", 
type `OP_CLAMP`, and backward callback `_clamp_backward_fn`, preserving boundary context for 
gradient masking during backward propagation.
```

> Signature
```c
TensorEngine* _clamp(TensorEngine *t1, f64 High, f64 Low);
```

> Example
```c
TensorEngine *x = tensor((f64[]){-5.0, 0.5, 2.0, 10.0, E}, (int[]){4, N}, false);
set_requires_grad(x, true);

TensorEngine *y = _clamp(x, 5.0, 0.0);
p(y); // [0.000000, 0.500000, 2.000000, 5.000000]

backward(y);
p(x); // x->grad contains [0.000000, 1.000000, 1.000000, 0.000000]

free_tensor(x);
free_tensor(y);
```

---

# Autograd Engine & Graph Functions (`gradfn.cpp`)

## Autograd Lifecycle & Traversal

### set_requires_grad
```
The `set_requires_grad` function configures whether a tensor tracks gradients in the computational 
graph. When set to `true`, the tensor is designated as a leaf node (`is_leaf = true`) and will 
accumulate upstream gradients into its `grad` member during backward graph execution.
```

> Signature
```c
void set_requires_grad(struct TensorEngine *t, bool requires_grad);
```

> Example
```c
TensorEngine *t = tensor((f64[]){1.0, 2.0, E}, (int[]){2, N}, false);
set_requires_grad(t, true);
```

---

### zero_grad
```
The `zero_grad` function clears and frees any accumulated gradients currently stored in `t->grad`, 
resetting the gradient pointer to `nullptr`. This prepares the tensor for a subsequent backward pass.
```

> Signature
```c
void zero_grad(struct TensorEngine *t);
```

> Example
```c
zero_grad(t);
```

---

### backward
```
The `backward` function performs reverse-mode automatic differentiation starting from the given `root` 
tensor. It performs a post-order depth-first search (DFS) topological sort over the computational DAG, 
seeds the root gradient with 1.0 (if `grad_output` is `nullptr`), and evaluates each node's backward 
`apply` callback in reverse topological order (from outputs to inputs). Gradients are accumulated into 
the `grad` buffer of saved operand tensors.
```

> Signature
```c
void backward(struct TensorEngine *root, struct TensorEngine *grad_output = nullptr);
```

> Example
```c
TensorEngine *a = tensor((f64[]){2.0, E}, (int[]){1, N}, false);
set_requires_grad(a, true);

TensorEngine *b = mlt(a, a); // b = a^2
backward(b);

p(a); // a->grad is 2*a = [4.000000]
```

---

### print_graph
```
The `print_graph` function recursively traverses and visualizes the Directed Acyclic Graph (DAG) 
originating at the specified tensor node. It prints operation node names (e.g. `-> AddBackward`, 
`-> MulBackward`), tensor shapes, and leaf status with indentation representing graph depth.
```

> Signature
```c
void print_graph(struct TensorEngine *t);
```

> Example
```c
TensorEngine *z = add(mlt(x, y), c);
print_graph(z);
```

---

## Graph Node Construction & Helper Functions

### create_grad_fn
```
The `create_grad_fn` function allocates and initializes a new `GradFn` computational graph node. 
It records the operator name, operation enum `GradFnType`, backward execution callback pointer `apply`, 
an array of saved input tensor references `saved_tensors`, and an optional operation-specific context 
pointer along with its cleanup destructor callback `release_context`.
```

> Signature
```c
GradFn* create_grad_fn(
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, struct TensorEngine *grad_output),
    size_t num_saved,
    struct TensorEngine **saved_tensors,
    void *context,
    void (*release_context)(void *context)
);
```

> Example
```c
TensorEngine *saved[1] = { input };
GradFn *fn = create_grad_fn("CustomOp", OP_CUSTOM, custom_bwd, 1, saved, NULL, NULL);
free_grad_fn(fn);
```

---

### free_grad_fn
```
The `free_grad_fn` function deallocates memory associated with a `GradFn` node. If a custom context 
destructor `release_context` is present, it is invoked to free context resources. It also frees the 
`saved_tensors` pointer array and the `GradFn` struct itself.
```

> Signature
```c
void free_grad_fn(GradFn *fn);
```

> Example
```c
free_grad_fn(node->grad_fn);
```

---

### accumulate_grad
```
The `accumulate_grad` function accumulates an incoming gradient tensor `incoming` into `target->grad`. 
If `target->grad` is initially `NULL`, it creates a standalone clone of `incoming`. If `target->grad` 
already exists, it adds `incoming` element-wise (using standard `add` if shapes match, or `add_broad` 
if broadcasting is required) and replaces `target->grad` with the new accumulated gradient sum.
```

> Signature
```c
void accumulate_grad(struct TensorEngine *target, struct TensorEngine *incoming);
```

> Example
```c
accumulate_grad(leaf_tensor, upstream_gradient);
```

---

### unbroadcast_to
```
The `unbroadcast_to` function performs gradient reduction across dimensions that were expanded or 
broadcasted during forward execution. It sums gradient values along padded leading dimensions and 
singular axes (where `target_shape[d] == 1`), returning a newly allocated tensor matching the exact 
target shape and dimensions.
```

> Signature
```c
struct TensorEngine* unbroadcast_to(struct TensorEngine *grad, const int *target_shape, size_t target_ndim);
```

> Example
```c
int orig_shape[] = {1, 4, N};
TensorEngine *reduced = unbroadcast_to(broad_grad, orig_shape, 2);
free_tensor(reduced);
```

---

## Graph Attachment Functions

### attach_binary_grad_fn
```
The `attach_binary_grad_fn` helper creates and attaches a two-input `GradFn` node to `result`, 
saving references to operands `t1` and `t2`. It sets `result->requires_grad = true` and 
`result->is_leaf = false`.
```

> Signature
```c
void attach_binary_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *t1,
    struct TensorEngine *t2,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, struct TensorEngine *grad_output)
);
```

> Example
```c
attach_binary_grad_fn(res, a, b, "AddBackward", OP_ADD, add_backward_fn);
```

---

### attach_unary_grad_fn
```
The `attach_unary_grad_fn` helper creates and attaches a single-input `GradFn` node to `result`, 
saving input tensor `input` and attaching optional context metadata along with its release destructor.
```

> Signature
```c
void attach_unary_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const char *name,
    GradFnType type,
    void (*apply)(GradFn *self, struct TensorEngine *grad_output),
    void *context,
    void (*release_context)(void *ctx)
);
```

> Example
```c
attach_unary_grad_fn(res, input, "CosBackward", OP_COS, _cos_backward_fn, NULL, NULL);
```

---

### attach_transpose_grad_fn
```
The `attach_transpose_grad_fn` helper packages transposition axes into a `TransposeCtx` structure 
and attaches a `TransposeBackward` (`OP_TRANSPOSE`) node to `result`.
```

> Signature
```c
void attach_transpose_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const int *axes
);
```

> Example
```c
int axes[] = {1, 0, N};
attach_transpose_grad_fn(transposed, orig, axes);
```

---

### attach_reshape_grad_fn
```
The `attach_reshape_grad_fn` helper packages the original shape and dimension count into a `ReshapeCtx` 
and attaches a `ReshapeBackward` (`OP_RESHAPE`) node to `result`.
```

> Signature
```c
void attach_reshape_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const int *orig_shape,
    size_t orig_ndim
);
```

> Example
```c
attach_reshape_grad_fn(reshaped, orig, orig->shape, orig->ndim);
```

---

### attach_slice_grad_fn
```
The `attach_slice_grad_fn` helper packages slicing coordinates (`starts`, `steps`, `out_shape`, and 
`offset`) into a `SliceCtx` and attaches a `SliceBackward` (`OP_SLICE`) node to `result`.
```

> Signature
```c
void attach_slice_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    const int *starts,
    const int *steps,
    const int *out_shape,
    int offset
);
```

> Example
```c
attach_slice_grad_fn(sliced, orig, starts, steps, out_shape, offset);
```

---

### attach_clamp_grad_fn
```
The `attach_clamp_grad_fn` helper allocates a `ClampCtx` storing lower and upper clamp bounds, 
attaching a `ClampBackward` (`OP_CLAMP`) node to `result`.
```

> Signature
```c
void attach_clamp_grad_fn(
    struct TensorEngine *result,
    struct TensorEngine *input,
    f64 low,
    f64 high
);
```

> Example
```c
attach_clamp_grad_fn(clamped, orig, 0.0, 1.0);
```

---

## Backward Functions

### add_backward_fn
```
The `add_backward_fn` callback evaluates the backward pass for element-wise addition `c = a + b`. 
It propagates incoming gradient `grad_output` unchanged to operands `a` and `b`:
  da = dc
  db = dc
```

> Signature
```c
void add_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### sub_backward_fn
```
The `sub_backward_fn` callback evaluates the backward pass for element-wise subtraction `c = a - b`:
  da = dc
  db = -dc
```

> Signature
```c
void sub_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### mul_backward_fn
```
The `mul_backward_fn` callback evaluates the backward pass for element-wise multiplication `c = a * b`:
  da = dc * b
  db = dc * a
```

> Signature
```c
void mul_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### div_backward_fn
```
The `div_backward_fn` callback evaluates the backward pass for element-wise division `c = a / b`:
  da = dc / b
  db = -dc * a / (b^2)
```

> Signature
```c
void div_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### _cos_backward_fn
```
The `_cos_backward_fn` callback computes the derivative of element-wise cosine `c = cos(a)`:
  da = grad_output * (-sin(a))
Gradients are accumulated into `a->grad`.
```

> Signature
```c
void _cos_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### _sin_backward_fn
```
The `_sin_backward_fn` callback computes the derivative of element-wise sine `c = sin(a)`:
  da = grad_output * cos(a)
Gradients are accumulated into `a->grad`.
```

> Signature
```c
void _sin_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### _clamp_backward_fn
```
The `_clamp_backward_fn` callback computes the derivative of element-wise clamping `c = clamp(a, low, high)`:
  da_i = grad_output_i  if  low <= a_i <= high  else  0.0
Boundary information is retrieved from `self->context`.
```

> Signature
```c
void _clamp_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### add_broad_backward_fn
```
The `add_broad_backward_fn` callback computes gradients for broadcasted addition `c = a + b`, 
unbroadcasting incoming gradients along expanded dimensions to match the shapes of `a` and `b`:
  da = unbroadcast(dc, a.shape)
  db = unbroadcast(dc, b.shape)
```

> Signature
```c
void add_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### sub_broad_backward_fn
```
The `sub_broad_backward_fn` callback computes gradients for broadcasted subtraction `c = a - b`:
  da = unbroadcast(dc, a.shape)
  db = unbroadcast(-dc, b.shape)
```

> Signature
```c
void sub_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### mul_broad_backward_fn
```
The `mul_broad_backward_fn` callback computes gradients for broadcasted multiplication `c = a * b`:
  da = unbroadcast(dc * b, a.shape)
  db = unbroadcast(dc * a, b.shape)
```

> Signature
```c
void mul_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### div_broad_backward_fn
```
The `div_broad_backward_fn` callback computes gradients for broadcasted division `c = a / b`:
  da = unbroadcast(dc / b, a.shape)
  db = unbroadcast(-dc * a / (b^2), b.shape)
```

> Signature
```c
void div_broad_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### dot_backward_fn
```
The `dot_backward_fn` callback evaluates the matrix multiplication backward pass for `C = A @ B`:
  dA = dC @ B^T
  dB = A^T @ dC
Handles 1D vector products, 1D-2D outer products, and general 2D/ND matrix dot products.
```

> Signature
```c
void dot_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### transpose_backward_fn
```
The `transpose_backward_fn` callback inverts the permutation of axes applied during forward transposition, 
transposing `grad_output` back to the original layout of the input tensor.
```

> Signature
```c
void transpose_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### reshape_backward_fn
```
The `reshape_backward_fn` callback reshapes `grad_output` back to the original shape and dimension 
count saved in `self->context`.
```

> Signature
```c
void reshape_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```

---

### slice_backward_fn
```
The `slice_backward_fn` callback constructs a zero-filled tensor matching the unsliced input shape and 
scatters `grad_output` elements back to their respective multi-dimensional indices according to the 
slice coordinates stored in `self->context`.
```

> Signature
```c
void slice_backward_fn(GradFn *self, struct TensorEngine *grad_output);
```
