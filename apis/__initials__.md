# Tensor Engine

## tensor
```
The `tensor` function is a primary function. You can use it to create a tensor
of any shape on either the CPU or the GPU. Note that you cannot simply pass 
the raw array and shape array directly; there are specific rules to follow. 
First, you must create a linear array of the `f64` data type (representing a 
double-precision float) and a separate shape array; ensure that the product of 
the shape elements matches the total number of elements in the data array. The 
third argument is a boolean: if set to `true`, the tensor is created on the 
GPU, whereas if set to `false`, it is created on the CPU. Keep in mind that 
the GPU is not immediately involved during the tensor's creation; the engine 
merely records that the tensor is intended for the GPU. GPU interaction 
actually occurs when an operation is performed on the tensor, at which point 
all operations are executed on the GPU. Finally, note that 'E' represents 
negative infinity (`-INFINITY`) and 'N' represents the minimum integer value 
(`INT_MIN`); these serve as safety guards.
```

> Signature
```
TensorEngine* tensor(f64 array_[], int shape_[], bool __GPU__);
```

> Example
```
f64 array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, E};
int shape[] = {2, 3, 2, N};
TensorEngine* t1  = tensor(array, shape, true);
p(t1);
free_tensor(t1);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){1, 2, 3, 5, 6, E}, (int[]){1, 6, N}, false);

p(t1);
free_tensor(t1);
```

## p
```
The `p` function prints the complete metadata and multidimensional contents of 
a `TensorEngine` instance to the standard output. It recursively formats the 
tensor data into nested brackets reflecting its multidimensional shape and 
strides. In addition to displaying the contents, `p` prints the total element 
`size`, number of dimensions (`ndim`), `shape` array, `strides` array, and the 
`__gpu__` execution target flag. If the provided tensor pointer is `NULL` or 
contains an invalid data buffer, an error message is printed to `stderr`.
```

> Signature
```
void p(TensorEngine *tensor);
```

> Example
```
f64 data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int shape[] = {2, 3, N};
TensorEngine* t = tensor(data, shape, false);

p(t);

free_tensor(t);
```

> Second Example
```
TensorEngine* t = arange(1.0, 5.0, 1.0, false);
p(t);
free_tensor(t);
```

## free_tensor
```
The `free_tensor` function releases all heap-allocated memory associated with a 
`TensorEngine` instance. This includes the underlying floating-point data buffer 
(`tensor`), the `shape` array, the `strides` array, and the `TensorEngine` 
struct itself. If a `nullptr` is passed, the function returns safely without 
performing any action. Always call `free_tensor` when a tensor is no longer 
needed to prevent memory leaks.
```

> Signature
```
void free_tensor(TensorEngine *t);
```

> Example
```
f64 array[] = {10.0, 20.0, 30.0, E};
int shape[] = {3, N};
TensorEngine* t = tensor(array, shape, false);

p(t);
free_tensor(t);
```

> Second Example
```
TensorEngine* t1 = arange(0.0, 10.0, 1.0, false);
TensorEngine* t2 = arange(10.0, 20.0, 1.0, false);
TensorEngine* res = add(t1, t2);

free_tensor(t1);
free_tensor(t2);
free_tensor(res);
```

## add
```
The `add` function performs element-wise addition between two tensors (`t1 + t2`) 
that have identical shapes. Both tensors must possess the exact same number of 
dimensions (`ndim`) and matching dimension lengths; otherwise, an error is 
reported to `stderr` and `NULL` is returned. If both tensors have their `__GPU__` 
flag set to `true`, the addition is executed on the GPU using a CUDA kernel; 
otherwise, it is computed on the CPU. A newly allocated `TensorEngine` pointer 
containing the result is returned.
```

> Signature
```
TensorEngine* add(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, E};
f64 b[] = {10.0, 20.0, 30.0, 40.0, E};
int shape[] = {2, 2, N};

TensorEngine* t1 = tensor(a, shape, false);
TensorEngine* t2 = tensor(b, shape, false);

TensorEngine* result = add(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){1.0, 2.0, 3.0, E}, (int[]){3, N}, true);
TensorEngine* t2 = tensor((f64[]){4.0, 5.0, 6.0, E}, (int[]){3, N}, true);

TensorEngine* result = add(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## sub
```
The `sub` function computes the element-wise subtraction of two tensors (`t1 - t2`). 
Both input tensors must have identical shapes and dimensions. If the shapes 
differ, an error is printed to `stderr` and `NULL` is returned. When both `t1` 
and `t2` have the `__GPU__` flag set to `true`, the operation is accelerated 
on the GPU via CUDA kernels; otherwise, computation runs on the CPU. Returns a 
newly created `TensorEngine` containing the difference values.
```

> Signature
```
TensorEngine* sub(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {10.0, 20.0, 30.0, 40.0, E};
f64 b[] = {1.0, 2.0, 3.0, 4.0, E};
int shape[] = {2, 2, N};

TensorEngine* t1 = tensor(a, shape, false);
TensorEngine* t2 = tensor(b, shape, false);

TensorEngine* result = sub(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){5.0, 8.0, 12.0, E}, (int[]){3, N}, true);
TensorEngine* t2 = tensor((f64[]){2.0, 3.0, 4.0, E}, (int[]){3, N}, true);

TensorEngine* result = sub(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## mlt
```
The `mlt` function performs element-wise multiplication (Hadamard product) 
between two tensors (`t1 * t2`). Both tensors must have matching shapes and 
dimension counts. If shape verification fails, `NULL` is returned. If both 
operands are configured for the GPU (`__GPU__ = true`), the operation is 
executed in parallel on the GPU; otherwise, standard CPU execution is used. 
Returns a new `TensorEngine` containing the multiplied values.
```

> Signature
```
TensorEngine* mlt(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {2.0, 3.0, 4.0, 5.0, E};
f64 b[] = {10.0, 20.0, 30.0, 40.0, E};
int shape[] = {2, 2, N};

TensorEngine* t1 = tensor(a, shape, false);
TensorEngine* t2 = tensor(b, shape, false);

TensorEngine* result = mlt(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){2.0, 4.0, 6.0, E}, (int[]){3, N}, true);
TensorEngine* t2 = tensor((f64[]){3.0, 2.0, 0.5, E}, (int[]){3, N}, true);

TensorEngine* result = mlt(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## divt
```
The `divt` function performs element-wise division of two tensors (`t1 / t2`). 
Both tensors must have identical shapes. The function includes built-in safety 
guards against division by zero: if a divisor element in `t2` is `0.0`, the 
resulting element is safely set to `0.0` instead of producing undefined behavior 
or NaN. When both tensors are targeted for GPU execution, the calculation runs 
on the GPU; otherwise, it executes on the CPU. Returns a new `TensorEngine` pointer.
```

> Signature
```
TensorEngine* divt(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {10.0, 20.0, 30.0, 40.0, E};
f64 b[] = {2.0, 4.0, 5.0, 8.0, E};
int shape[] = {2, 2, N};

TensorEngine* t1 = tensor(a, shape, false);
TensorEngine* t2 = tensor(b, shape, false);

TensorEngine* result = divt(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
// Demonstrating division-by-zero protection (divisor 0.0 results in 0.0)
TensorEngine* t1 = tensor((f64[]){10.0, 20.0, 30.0, E}, (int[]){3, N}, false);
TensorEngine* t2 = tensor((f64[]){2.0, 0.0, 5.0, E}, (int[]){3, N}, false);

TensorEngine* result = divt(t1, t2);
p(result); // result[1] will be 0.000000

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## mod
```
The `mod` function computes the element-wise floating-point modulus / remainder 
of two tensors (`fmod(t1, t2)`). Both tensors must have matching shapes. The 
function includes division-by-zero protection: if a divisor element in `t2` is 
`0.0`, the result for that element is set to `0.0`. This operation executes on 
the CPU and returns a newly allocated `TensorEngine` pointer.
```

> Signature
```
TensorEngine* mod(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {7.0, 10.0, 15.5, 20.0, E};
f64 b[] = {3.0, 4.0, 4.0, 6.0, E};
int shape[] = {2, 2, N};

TensorEngine* t1 = tensor(a, shape, false);
TensorEngine* t2 = tensor(b, shape, false);

TensorEngine* result = mod(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){5.5, 9.0, 11.2, E}, (int[]){3, N}, false);
TensorEngine* t2 = tensor((f64[]){2.0, 3.0, 4.0, E}, (int[]){3, N}, false);

TensorEngine* result = mod(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## add_broad
```
The `add_broad` function computes the broadcast element-wise addition of two 
tensors (`t1 + t2`) following NumPy-style broadcasting rules. When shapes differ, 
dimensions are aligned from right to left, and singleton dimensions (size 1) or 
missing leading dimensions are automatically expanded to match the output shape. 
If the shapes are incompatible for broadcasting, `NULL` is returned. If both 
tensors are on the GPU, a specialized broadcast CUDA kernel executes the 
operation; otherwise, it runs on the CPU.
```

> Signature
```
TensorEngine* add_broad(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int s1[] = {2, 3, N};
TensorEngine* t1 = tensor(a, s1, false);

f64 b[] = {10.0, 20.0, 30.0, E};
int s2[] = {1, 3, N};
TensorEngine* t2 = tensor(b, s2, false);

TensorEngine* result = add_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){1.0, 2.0, E}, (int[]){2, 1, N}, true);
TensorEngine* t2 = tensor((f64[]){10.0, 20.0, 30.0, E}, (int[]){3, N}, true);

TensorEngine* result = add_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## sub_broad
```
The `sub_broad` function performs broadcast element-wise subtraction (`t1 - t2`) 
between two tensors according to standard broadcasting rules. Singleton and 
leading dimensions are stretched to match the broadcast output shape. Returns 
`NULL` if the shapes cannot be broadcast together. Supports both CPU and GPU 
execution depending on whether both input tensors have the `__GPU__` flag set.
```

> Signature
```
TensorEngine* sub_broad(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, E};
int s1[] = {2, 3, N};
TensorEngine* t1 = tensor(a, s1, false);

f64 b[] = {1.0, 2.0, 3.0, E};
int s2[] = {1, 3, N};
TensorEngine* t2 = tensor(b, s2, false);

TensorEngine* result = sub_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){50.0, 100.0, E}, (int[]){2, 1, N}, true);
TensorEngine* t2 = tensor((f64[]){5.0, 10.0, 15.0, E}, (int[]){3, N}, true);

TensorEngine* result = sub_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## mlt_broad
```
The `mlt_broad` function executes broadcast element-wise multiplication 
(`t1 * t2`) between two tensors. Compatible dimensions with size 1 or missing 
leading dimensions are broadcast across the target tensor shape. If shapes are 
incompatible, `NULL` is returned. If both tensors reside on the GPU, the 
computation is dispatched via a broadcast CUDA kernel. Returns the resulting 
`TensorEngine` pointer.
```

> Signature
```
TensorEngine* mlt_broad(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int s1[] = {2, 3, N};
TensorEngine* t1 = tensor(a, s1, false);

f64 b[] = {2.0, 3.0, 4.0, E};
int s2[] = {3, N};
TensorEngine* t2 = tensor(b, s2, false);

TensorEngine* result = mlt_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){2.0, 3.0, E}, (int[]){2, 1, N}, true);
TensorEngine* t2 = tensor((f64[]){10.0, 20.0, 30.0, E}, (int[]){1, 3, N}, true);

TensorEngine* result = mlt_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## div_broad
```
The `div_broad` function computes broadcast element-wise division (`t1 / t2`). 
Tensors are broadcast to a common shape, and any division by `0.0` is safely 
handled by returning `0.0` for that element. If the dimensions are incompatible 
for broadcasting, `NULL` is returned. Supports both GPU execution (when both 
operands have `__GPU__ = true`) and CPU execution.
```

> Signature
```
TensorEngine* div_broad(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, E};
int s1[] = {2, 3, N};
TensorEngine* t1 = tensor(a, s1, false);

f64 b[] = {2.0, 5.0, 10.0, E};
int s2[] = {3, N};
TensorEngine* t2 = tensor(b, s2, false);

TensorEngine* result = div_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){100.0, 200.0, E}, (int[]){2, 1, N}, true);
TensorEngine* t2 = tensor((f64[]){2.0, 4.0, 0.0, E}, (int[]){1, 3, N}, true);

TensorEngine* result = div_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## mod_broad
```
The `mod_broad` function computes broadcast floating-point modulus / remainder 
(`fmod(t1, t2)`). The inputs are broadcast to a common compatible shape. Division 
by `0.0` produces `0.0` safely. The function supports GPU acceleration when 
both tensors have `__GPU__` set to `true`, as well as CPU execution. Returns a 
new `TensorEngine` pointer.
```

> Signature
```
TensorEngine* mod_broad(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, E};
int s1[] = {2, 3, N};
TensorEngine* t1 = tensor(a, s1, false);

f64 b[] = {3.0, 4.0, 5.0, E};
int s2[] = {3, N};
TensorEngine* t2 = tensor(b, s2, false);

TensorEngine* result = mod_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* t1 = tensor((f64[]){15.0, 25.0, E}, (int[]){2, 1, N}, true);
TensorEngine* t2 = tensor((f64[]){4.0, 7.0, 10.0, E}, (int[]){1, 3, N}, true);

TensorEngine* result = mod_broad(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

## dot_prod
```
The `dot_prod` function computes the tensor dot product or matrix multiplication 
between two tensors (`t1` and `t2`). It contracts the last dimension of `t1` 
with the second-to-last dimension (or the only dimension if 1D) of `t2`. 
- For two 1D vectors of length K, it computes the inner product, returning a 
  tensor of shape `[1]`.
- For two 2D matrices of shapes `[M, K]` and `[K, P]`, it computes matrix 
  multiplication, returning a tensor of shape `[M, P]`.
- For multidimensional tensors, higher dimensions are batched and preserved 
  accordingly.
If the contracted inner dimensions do not match, `NULL` is returned. If both 
tensors are on the GPU (`__GPU__ = true`), the operation is accelerated using 
CUDA matrix multiplication kernels; otherwise, it is computed on the CPU.
```

> Signature
```
TensorEngine* dot_prod(TensorEngine* t1, TensorEngine* t2);
```

> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int s1[] = {2, 3, N};
TensorEngine* t1 = tensor(a, s1, false);

f64 b[] = {7.0, 8.0, 9.0, 1.0, 2.0, 3.0, E};
int s2[] = {3, 2, N};
TensorEngine* t2 = tensor(b, s2, false);

TensorEngine* result = dot_prod(t1, t2);
p(result);

free_tensor(t1);
free_tensor(t2);
free_tensor(result);
```

> Second Example
```
TensorEngine* v1 = tensor((f64[]){1.0, 2.0, 3.0, E}, (int[]){3, N}, true);
TensorEngine* v2 = tensor((f64[]){4.0, 5.0, 6.0, E}, (int[]){3, N}, true);

TensorEngine* result = dot_prod(v1, v2);
p(result);

free_tensor(v1);
free_tensor(v2);
free_tensor(result);
```

## T
```
The `T` function transposes a tensor. 
- For a 1D tensor with shape `[N]`, it reshapes the tensor into a 2D column 
  matrix with shape `[N, 1]`.
- For a 2D or multidimensional tensor, if `axes` is `nullptr`, it reverses all 
  axes (e.g., shape `[M, N]` becomes `[N, M]`, or shape `[D0, D1, D2]` becomes 
  `[D2, D1, D0]`).
- If a custom `axes` permutation array is provided, it permutes the dimensions 
  according to the specified axis ordering (e.g., passing `{1, 0, 2}` swaps the 
  first two axes of a 3D tensor).
Supports both CPU and GPU execution.
```

> Signature
```
TensorEngine* T(TensorEngine *t, int *axes = nullptr);
```

> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, E};
int shape[] = {2, 3, N};
TensorEngine* t = tensor(a, shape, false);

TensorEngine* transposed = T(t);
p(transposed);

free_tensor(t);
free_tensor(transposed);
```

> Second Example
```
TensorEngine* t = arange(1.0, 25.0, 1.0, false);
int shape3d[] = {2, 3, 4};
t = reshape(t, shape3d, 3);

int axes[] = {2, 0, 1};
TensorEngine* transposed = T(t, axes);
p(transposed);

free_tensor(t);
free_tensor(transposed);
```

## transpose
```
The `transpose` function is an alias for the `T` function. It transposes 1D, 
2D, or multidimensional tensors with optional custom axis permutations. Passing 
`nullptr` as the `axes` argument reverses all dimensions by default. Supports 
both CPU and GPU execution.
```

> Signature
```
TensorEngine* transpose(TensorEngine *t, int *axes = nullptr);
```

> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, E};
int shape[] = {2, 2, N};
TensorEngine* t = tensor(a, shape, false);

TensorEngine* res = transpose(t);
p(res);

free_tensor(t);
free_tensor(res);
```

> Second Example
```
TensorEngine* t = arange(1.0, 5.0, 1.0, false);

TensorEngine* col = transpose(t);
p(col);

free_tensor(t);
free_tensor(col);
```

## extract
```
The `extract` function squeezes off *leading* dimensions of size `1` from a 
tensor's shape, without altering the underlying data or its element order. 
It does not flatten the tensor into 1D — trailing dimensions are preserved 
as-is; only leading `1`-sized dimensions are dropped.
- The second argument `expected_size` is a safety check: it must equal the 
  tensor's total element count (`t1->size`), or `NULL` is returned with an 
  error. It is not used for any computation — pass `t1->size` directly to 
  avoid having to compute it by hand.
- If every dimension in the tensor is `1`, one dimension is preserved so the 
  result is never a 0-dimensional tensor.
- Since leading `1`-sized dimensions never affect memory layout in row-major 
  order, the underlying data is copied unchanged; only the shape metadata 
  is rebuilt.
Supports both CPU and GPU execution.

extract method is frequently used after the slice method
```
 
> Signature
```
TensorEngine* extract(TensorEngine *t1, int expected_size);
```
 
> Example
```
f64 a[] = {1.0, 2.0, 3.0, 4.0, E};
int shape[] = {1, 1, 1, 4, N};
TensorEngine* t = tensor(a, shape, false);
 
TensorEngine* res = extract(t, t->size);
p(res);
 
free_tensor(t);
free_tensor(res);
```
 
> Second Example
```
TensorEngine* t = arange(1.0, 9.0, 1.0, false);
int shape[] = {1, 1, 1, 2, 2, 2, N};
t = reshape(t, shape, 6);
 
TensorEngine* res = extract(t, t->size);
p(res);
 
free_tensor(t);
free_tensor(res);
```

## arange
```
The `arange` function creates a 1D tensor containing evenly spaced values within 
the half-open interval `[start, end)` with a given `step` increment. 
- The number of elements generated is `ceil((end - start) / step)`.
- The `step` value must not be `0`, and the step direction (positive or negative) 
  must match the start/end bounds; otherwise, `NULL` is returned.
- The fourth argument `gpu` is a boolean indicating whether the tensor is intended 
  for GPU or CPU execution.
Returns a newly allocated 1D `TensorEngine` pointer.
```

> Signature
```
TensorEngine* arange(f64 start, f64 end, f64 step, bool gpu);
```

> Example
```
TensorEngine* t = arange(0.0, 10.0, 2.0, false);
p(t);
free_tensor(t);
```

> Second Example
```
TensorEngine* t = arange(10.0, 0.0, -2.0, true);
p(t);
free_tensor(t);
```

## linspace
```
The `linspace` function generates a 1D tensor containing `num` evenly spaced 
values calculated over the closed interval `[start, end]`. Both endpoints 
(`start` and `end`) are included in the generated sequence (when `num > 1`). 
The parameter `num` must be greater than `0`. The fourth argument `__GPU__` 
specifies whether the tensor is created for GPU or CPU execution. Returns a newly 
allocated 1D `TensorEngine` pointer.
```

> Signature
```
TensorEngine* linspace(f64 start, f64 end, int num, bool __GPU__);
```

> Example
```
TensorEngine* t = linspace(0.0, 1.0, 5, false);
p(t);
free_tensor(t);
```

> Second Example
```
TensorEngine* t = linspace(10.0, 100.0, 10, true);
p(t);
free_tensor(t);
```

## reshape
```
The `reshape` function returns a new tensor with the specified `new_shape` and 
number of dimensions `new_ndim`, without altering the underlying data values. 
The total number of elements in `new_shape` (product of all dimensions) must 
exactly equal the original tensor's total size (`t->size`); otherwise, an error 
is printed and `NULL` is returned. All dimension values in `new_shape` must be 
positive integers. The `__GPU__` flag of the original tensor is preserved in 
the newly returned tensor.
```

> Signature
```
TensorEngine* reshape(TensorEngine* t, int *new_shape, size_t new_ndim);
```

> Example
```
TensorEngine* t = arange(1.0, 7.0, 1.0, false);

int new_shape[] = {2, 3};
TensorEngine* reshaped = reshape(t, new_shape, 2);
p(reshaped);

free_tensor(t);
free_tensor(reshaped);
```

> Second Example
```
TensorEngine* t = arange(1.0, 25.0, 1.0, true);

int shape3d[] = {2, 3, 4};
TensorEngine* tensor3d = reshape(t, shape3d, 3);
p(tensor3d);

free_tensor(t);
free_tensor(tensor3d);
```

## slice
```
The `slice` function extracts a sub-tensor from an existing multidimensional 
tensor `t` using multidimensional slicing specifications. 
- The `indices` argument is a 2D array where each row represents `{start, stop, step}` 
  for the corresponding dimension.
- Special sentinel `N` (`INT_MIN`) can be passed to denote default slice boundaries:
  - `start = N`: defaults to `0` (for positive step) or `dim_len - 1` (for negative step).
  - `stop = N`: defaults to `dim_len` (for positive step) or `-1` (for negative step).
  - `step = N`: defaults to `1`.
- Negative indexing is fully supported (e.g., `-1` references the last element).
- Negative step values are supported for reversing or striding backwards.
- `num_slices` specifies how many dimensions to slice (defaults to `t->ndim` when set to `0`).
- `offset` applies an optional initial linear element offset (defaults to `0`).
Supports both CPU and GPU execution.
```

> Signature
```
TensorEngine* slice(TensorEngine* t, int indices[][3], size_t num_slices, int offset);
```

> Example
```
TensorEngine* t = arange(1.0, 13.0, 1.0, false);
int shape[] = {3, 4};
TensorEngine* mat = reshape(t, shape, 2, 0);

int s[][3] = {
    {0, 2, 1}, // Rows: [0, 2] step 1
    {1, 4, 2}  // Cols: [1, 4] step 2
};
TensorEngine* sub_mat = slice(mat, s, 2, 0);
// 0 is offset, you have to always pass offset as 0
p(sub_mat);

free_tensor(t);
free_tensor(mat);
free_tensor(sub_mat);
```

> Second Example
```
TensorEngine* tx = arange(1.0, 65.0, 1.0, false);
int shape6d[] = {2, 2, 2, 2, 2, 2};
tx = reshape(tx, shape6d, 6);

int s6d[][3] = {
    {0, 1, 1}, // Dim 0: index 0
    {0, 1, 1}, // Dim 1: index 0
    {1, 2, 1}, // Dim 2: index 1
    {N, N, N}, // Dim 3: all elements (default)
    {N, N, N}, // Dim 4: all elements (default)
    {N, N, N}  // Dim 5: all elements (default)
};
TensorEngine* sl6d = slice(tx, s6d);
p(sl6d);

free_tensor(tx);
free_tensor(sl6d);
```
