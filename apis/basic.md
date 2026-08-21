
# Tensor Engine

## tensor
```
The `tensor` function is a primary function. You can use it to create a tensor of any shape on either the CPU or the GPU. Note that you cannot simply pass the raw array and shape array directly; there are specific rules to follow. First, you must create a linear array of the `f64` data type (representing a double-precision float) and a separate shape array; ensure that the product of the shape elements matches the total number of elements in the data array. The third argument is a boolean: if set to `true`, the tensor is created on the GPU, whereas if set to `false`, it is created on the CPU. Keep in mind that the GPU is not immediately involved during the tensor's creation; the engine merely records that the tensor is intended for the GPU. GPU interaction actually occurs when an operation is performed on the tensor, at which point all operations are executed on the GPU. Finally, note that 'E' represents negative infinity (`-INFINITY`) and 'N' represents the minimum integer value (`INT_MIN`); these serve as safety guards.
```

> Example
```
f64 array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, E};
int shape[] = {2, 3, 2, N};
TensorEngine* t1  = tensor(array, shape, true);
p(t1);
free_tensor(t1);
```