## encode_image_as_tensor
```
The `_encode_image_as_tensor` function converts a raw `uint8_t` pixel buffer 
into a `TensorEngine` of shape `[H, W, C]`, where `H` is height, `W` is width, 
and `C` is the number of channels (e.g. `3` for RGB, `4` for RGBA, `1` for 
grayscale).
- `image_array` must contain exactly `H * W * C` bytes, laid out in row-major 
  order (height, then width, then channel) — the same layout most raw image 
  buffers already use.
- Each byte is copied and widened into an `f64` value in the resulting tensor; 
  no scaling or normalization is applied.
- The fourth argument `__GPU_` is a boolean indicating whether the tensor is 
  intended for GPU or CPU execution.
Returns a newly allocated `TensorEngine` pointer with shape `[H, W, C]`, or 
`NULL` if memory allocation fails.
```

> Signature
```
TensorEngine* _encode_image_as_tensor(uint8_t *image_array, size_t H, size_t W, size_t C, bool __GPU_);
```

> Example
```
uint8_t raw[48]; // 4x4 RGB image, flattened: 4*4*3 = 48 bytes
for (int i = 0; i < 48; i++) raw[i] = (uint8_t)(i % 256);

TensorEngine* img = _encode_image_as_tensor(raw, 4, 4, 3, false);
p(img);

free_tensor(img);
```

> Second Example
```
uint8_t gray[64]; // 8x8 grayscale image: 8*8*1 = 64 bytes
for (int i = 0; i < 64; i++) gray[i] = (uint8_t)(i * 4);

TensorEngine* img = _encode_image_as_tensor(gray, 8, 8, 1, true);
p(img);

free_tensor(img);
```

> Example Three
```
uint8_t image[5][5][3] = {
        {
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {255, 255,   0},
            {255, 255, 255}
        },
        {
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {255, 255,   0},
            {255, 255, 255}
        },
        {
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {255, 255,   0},
            {255, 255, 255}
        },
        {
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {255, 255,   0},
            {255, 255, 255}
        },
        {
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {255, 255,   0},
            {255, 255, 255}
        }
    };

TensorEngine* t1 = _encode_image_as_tensor(&image[0][0][0], 5, 5, 3, false);
p(t1);
free_tensor(t1);
```