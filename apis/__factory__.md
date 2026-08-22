# Tensor Engine

## ones
```
The `ones` function creates and returns a new tensor with the specified `shape` 
where all elements are initialized to 1.0. To define the tensor's multidimensional 
dimensions, pass an integer array terminated by the sentinel `N` (`INT_MIN`). 
The second argument is a boolean `__GPU__`: when set to `true`, the tensor is 
configured for GPU execution; when set to `false`, it executes on the CPU. The 
function computes total element count (`size`), number of dimensions (`ndim`), 
and `strides` automatically, allocates heap memory for the `TensorEngine` struct, 
its shape array, and its `f64` data buffer, fills all data elements with `1.0`, 
and appends the terminal guard sentinel `E` (`-INFINITY`). If memory allocation 
fails, an error message is printed to `stderr` and `NULL` is returned. Always 
release allocated tensors with `free_tensor` when no longer needed.
```

> Signature
```
TensorEngine* ones(int shape[], bool __GPU__);
```

> Example
```
int shape[] = {2, 3, 2, N};
TensorEngine* t1 = ones(shape, false);

p(t1);
free_tensor(t1);
```

> Second Example
```
TensorEngine* t1 = ones((int[]){3, 3, N}, true);

p(t1);
free_tensor(t1);
```

## zeros
```
The `zeros` function creates and returns a new tensor of the specified `shape` 
with all elements initialized to 0.0. The `shape` array must be terminated with 
the sentinel value `N` (`INT_MIN`). The second parameter `__GPU__` determines 
whether the tensor targets GPU (`true`) or CPU (`false`) execution. The function 
allocates the `TensorEngine` instance, duplicates the shape array, calculates 
`size`, `ndim`, and `strides`, initializes all data buffer elements to `0.0`, 
and appends the sentinel `E` (`-INFINITY`). If memory allocation fails at any 
point, an error message is printed to `stderr` and `NULL` is returned.
```

> Signature
```
TensorEngine* zeros(int shape[], bool __GPU__);
```

> Example
```
int shape[] = {2, 4, N};
TensorEngine* t1 = zeros(shape, false);

p(t1);
free_tensor(t1);
```

> Second Example
```
TensorEngine* t1 = zeros((int[]){2, 2, 5, N}, true);

p(t1);
free_tensor(t1);
```

## empty
```
The `empty` function creates and returns a new tensor of the specified `shape` 
without initializing its data buffer values. This function is faster than `zeros` 
or `ones` because it skips the element-filling loop (`skip_fill = true`), making 
it ideal for temporary allocation where values will be immediately overwritten by 
subsequent computations. The `shape` array must be terminated with the sentinel 
value `N` (`INT_MIN`). The `__GPU__` parameter sets the execution target device. 
The metadata fields (`size`, `ndim`, `strides`, `shape`) are fully computed and 
assigned, and the buffer is terminated with `E`. Returns `NULL` if allocation fails.
```

> Signature
```
TensorEngine* empty(int shape[], bool __GPU__);
```

> Example
```
int shape[] = {3, 3, N};
TensorEngine* t1 = empty(shape, false);

p(t1);
free_tensor(t1);
```

> Second Example
```
TensorEngine* t1 = empty((int[]){4, 2, N}, true);

p(t1);
free_tensor(t1);
```

## full
```
The `full` function creates and returns a new tensor of the specified `shape` 
with all elements initialized to the scalar value `which` of type `f64` (double-
precision float). The `shape` array must be terminated with the safety sentinel 
`N` (`INT_MIN`). The `which` parameter specifies the constant value assigned to 
every element in the tensor's data buffer. The `__GPU__` flag specifies whether 
the tensor is intended for GPU or CPU computation. The function calculates `ndim`, 
`size`, and `strides`, populates the buffer, and appends the sentinel `E`. If 
memory allocation fails, an error is printed to `stderr` and `NULL` is returned.
```

> Signature
```
TensorEngine* full(int shape[], f64 which, bool __GPU__);
```

> Example
```
int shape[] = {2, 3, N};
TensorEngine* t1 = full(shape, 7.5, false);

p(t1);
free_tensor(t1);
```

> Second Example
```
TensorEngine* t1 = full((int[]){2, 2, 5, N}, 55.0, true);

p(t1);
free_tensor(t1);
```

## ones_alike
```
The `ones_alike` function creates and returns a new tensor having the exact same 
shape and dimensions as an existing prototype tensor `t1`, with all elements 
initialized to 1.0. Instead of requiring an explicit shape array, the function 
automatically extracts and duplicates the `shape` array from `t1`. The `__GPU__` 
boolean parameter specifies the target device (GPU if `true`, CPU if `false`) for 
the new tensor, which can be configured independently of `t1`'s target device. 
If `t1` is `NULL` (`nullptr`), an error message is printed to `stderr` and `NULL` 
is returned.
```

> Signature
```
TensorEngine* ones_alike(TensorEngine* t1, bool __GPU__);
```

> Example
```
int shape[] = {2, 3, N};
TensorEngine* proto = zeros(shape, false);

TensorEngine* t1 = ones_alike(proto, false);
p(t1);

free_tensor(proto);
free_tensor(t1);
```

> Second Example
```
TensorEngine* proto = full((int[]){2, 2, 5, N}, 55.0, true);
TensorEngine* t1 = ones_alike(proto, false);

p(t1);

free_tensor(proto);
free_tensor(t1);
```

## zeros_alike
```
The `zeros_alike` function creates and returns a new tensor with the exact same 
shape and dimensionality as an existing prototype tensor `t1`, with all elements 
initialized to 0.0. The function clones the shape configuration and dimension 
metadata directly from `t1` and allocates a matching data buffer. The `__GPU__` 
argument specifies whether the newly created tensor executes on the GPU or CPU. 
If `t1` is `NULL` (`nullptr`), an error message is logged to `stderr` and `NULL` 
is returned.
```

> Signature
```
TensorEngine* zeros_alike(TensorEngine* t1, bool __GPU__);
```

> Example
```
int shape[] = {3, 2, N};
TensorEngine* proto = ones(shape, false);

TensorEngine* t1 = zeros_alike(proto, false);
p(t1);

free_tensor(proto);
free_tensor(t1);
```

> Second Example
```
TensorEngine* proto = full((int[]){2, 4, N}, 42.0, true);
TensorEngine* t1 = zeros_alike(proto, false);

p(t1);

free_tensor(proto);
free_tensor(t1);
```

## empty_alike
```
The `empty_alike` function creates and returns a new uninitialized tensor matching 
the shape and dimensions of an existing prototype tensor `t1`. Memory is allocated 
for the new data buffer without populating its values, providing high performance 
when the tensor will be filled immediately by subsequent operations. The shape 
and dimension metadata are cloned from `t1`. The `__GPU__` argument sets the 
intended target device. If `t1` is `NULL` (`nullptr`), an error message is output 
to `stderr` and `NULL` is returned.
```

> Signature
```
TensorEngine* empty_alike(TensorEngine* t1, bool __GPU__);
```

> Example
```
int shape[] = {2, 5, N};
TensorEngine* proto = full(shape, 3.14, false);

TensorEngine* t1 = empty_alike(proto, false);
p(t1);

free_tensor(proto);
free_tensor(t1);
```

> Second Example
```
TensorEngine* proto = arange(0.0, 12.0, 1.0, true);
int s[] = {3, 4};
TensorEngine* mat = reshape(proto, s, 2);

TensorEngine* t1 = empty_alike(mat, true);
p(t1);

free_tensor(proto);
free_tensor(mat);
free_tensor(t1);
```

## full_alike
```
The `full_alike` function creates and returns a new tensor with the exact same 
shape and dimensionality as an existing prototype tensor `t1`, with all elements 
initialized to the specified scalar value `which` of type `f64`. The shape and 
metadata are cloned from `t1`, and all elements in the newly allocated data buffer 
are populated with `which` before terminating with `E`. The `__GPU__` boolean flag 
specifies whether the tensor is intended for GPU or CPU computation. If `t1` is 
`NULL` (`nullptr`), an error is logged to `stderr` and `NULL` is returned.
```

> Signature
```
TensorEngine* full_alike(TensorEngine* t1, f64 which, bool __GPU__);
```

> Example
```
int shape[] = {2, 3, N};
TensorEngine* proto = zeros(shape, false);

TensorEngine* t1 = full_alike(proto, 99.0, false);
p(t1);

free_tensor(proto);
free_tensor(t1);
```

> Second Example
```
TensorEngine* t1 = full((int[]){2, 2, 5, N}, 55.0, true);
TensorEngine* t2 = full_alike(t1, -1.0, false);

p(t2);

free_tensor(t1);
free_tensor(t2);
```
