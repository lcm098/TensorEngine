## random_seed
```
The `random_seed` function seeds the global pseudo-random number generator 
used internally by every `random_*` and `rand_*` function in this module. 
Calling it once at the start of a program makes all subsequent random tensor 
generation reproducible for a given seed value. If `random_seed` is never 
called explicitly, the generator self-seeds from the current system time on 
first use, producing a different sequence on each run.
```

> How it works
```
Internally, every random function in this module shares one global PRNG state 
built on C's standard rand()/srand(). random_seed(seed) calls srand(seed) 
directly, which resets rand()'s internal state so that every subsequent call 
to rand() follows a fixed, repeatable sequence for that seed. A flag 
(rng_seeded) is also set so that no other function overwrites this seed later 
with a time-based one. If random_seed is never called, the first random 
operation in the program calls srand(time(NULL)) automatically, seeding from 
the current clock time — this is why unseeded runs differ each execution, 
while seeded runs are exactly reproducible.
```

> Signature
```
void random_seed(unsigned int seed);
```

> Example
```
random_seed(42);

TensorEngine* t1 = random_randn((int[]){2, 3, N}, false);
p(t1);

free_tensor(t1);
```

> Second Example
```
random_seed(1234);
TensorEngine* a = random_uniform((int[]){4, N}, 0.0, 1.0, false);

random_seed(1234);
TensorEngine* b = random_uniform((int[]){4, N}, 0.0, 1.0, false);
// a and b will contain identical values, since the seed was reset

p(a);
p(b);

free_tensor(a);
free_tensor(b);
```

## random_shuffle
```
The `random_shuffle` function shuffles a tensor's elements in place using the 
Fisher-Yates algorithm. The tensor is treated as a flat sequence of `t->size` 
elements regardless of its actual shape — the shape metadata is left 
unchanged, only the element order is randomized. If the tensor resides on the 
GPU, its data is copied to the host, shuffled, and copied back automatically. 
Tensors with fewer than 2 elements are left unmodified. Passing `NULL` prints 
an error to `stderr`.
```

> How it works
```
random_shuffle implements the Fisher-Yates (Knuth) shuffle: it walks the 
tensor's flat data backward from the last index to the second index. At each 
position i, it picks a random index j uniformly from [0, i] (using rejection 
sampling to avoid modulo bias) and swaps element i with element j. This 
guarantees every possible ordering of the elements is equally likely, with 
no bias toward any particular permutation. If the tensor is on the GPU, its 
data is first copied to a temporary host buffer (since the shuffle swaps are 
done with simple host-side array indexing), shuffled in that buffer, then 
copied back to the device — the tensor's device pointer itself never changes, 
only the values it points to.
```

> Signature
```
void random_shuffle(TensorEngine *t);
```

> Example
```
TensorEngine* t = arange(1.0, 11.0, 1.0, false);

random_shuffle(t);
p(t);

free_tensor(t);
```

> Second Example
```
TensorEngine* t = arange(0.0, 12.0, 1.0, true);
int shape[] = {3, 4};
t = reshape(t, shape, 2);

random_shuffle(t); // shuffled as a flat 12-element sequence, shape stays [3, 4]
p(t);

free_tensor(t);
```

## rand_f64
```
The `rand_f64` function creates a new tensor of the given shape, filled with 
values uniformly distributed in the half-open interval `[low, high)`. This is 
the low-level uniform generator that `random_uniform` also delegates to. 
Supports both CPU and GPU tensor creation via the `__GPU__` flag.
```

> How it works
```
For each element in the output, rand_f64 first draws a raw value from rand() 
and divides it by (RAND_MAX + 1.0) to get a uniform double strictly within 
[0.0, 1.0). This normalized value u is then linearly mapped into the target 
range using low + u * (high - low). Because u never reaches exactly 1.0, the 
result never reaches high, which is why the interval is described as 
half-open: [low, high). All values are generated on the host first (since 
rand() is a CPU-only function); if __GPU__ is true, the completed host buffer 
is handed to the tensor constructor, which performs the host-to-device 
transfer during construction.
```

> Signature
```
TensorEngine* rand_f64(f64 low, f64 high, int shape[], bool __GPU__);
```

> Example
```
int shape[] = {3, 3, N};
TensorEngine* t = rand_f64(-5.0, 5.0, shape, false);
p(t);

free_tensor(t);
```

> Second Example
```
int shape[] = {2, 4, N};
TensorEngine* t = rand_f64(0.0, 100.0, shape, true);
p(t);

free_tensor(t);
```

## random_randn
```
The `random_randn` function creates a new tensor of the given shape, filled 
with values drawn from the standard normal distribution `N(0, 1)` (mean `0`, 
standard deviation `1`). It is a convenience wrapper equivalent to calling 
`random_normal(shape, 0.0, 1.0, __GPU__)`. Supports both CPU and GPU tensor 
creation.
```

> How it works
```
random_randn is a thin wrapper that calls random_normal(shape, 0.0, 1.0, 
__GPU__) directly — it does not implement its own sampling logic. See 
random_normal's "How it works" section below for the Box-Muller transform 
details that actually generate each value.
```

> Signature
```
TensorEngine* random_randn(int shape[], bool __GPU__);
```

> Example
```
int shape[] = {2, 3, N};
TensorEngine* t = random_randn(shape, false);
p(t);

free_tensor(t);
```

> Second Example
```
int shape[] = {5, N};
TensorEngine* t = random_randn(shape, true);
p(t);

free_tensor(t);
```

## random_choice
```
The `random_choice` function randomly selects `n` elements from `src` 
(treated as a flat sequence of `src->size` elements) and returns them as a new 
1D tensor of length `n`. 
- If `replace` is `true`, elements may be selected more than once.
- If `replace` is `false`, each selection is unique; `n` must not exceed 
  `src->size`, or `NULL` is returned with an error.
- If `src` resides on the GPU, its data is copied to the host for sampling, 
  and the result is built for the target device specified by `__GPU__`.
```

> How it works
```
If src lives on the GPU, its full data buffer is first copied to a temporary 
host array, since sampling itself is done with plain host-side indexing.

With replace = true: for each of the n output slots, a random index is drawn 
independently and uniformly from [0, src->size) (via rejection sampling to 
avoid modulo bias), and that source element is copied into the result. 
Because each draw is independent, the same source index can be selected more 
than once.

With replace = false: an internal index pool of size src->size is built, 
initially containing every valid index 0..src->size-1. For each of the n 
output slots, a random position within the remaining (shrinking) pool is 
picked, its corresponding source element is copied into the result, and that 
pool slot is then overwritten with the pool's current last element before 
shrinking the effective pool size by one (a partial Fisher-Yates over 
indices). This guarantees no index is ever picked twice, without needing to 
rescan for already-used indices on every draw.

The result is always built fresh as a new 1D tensor; src itself is never 
modified.
```

> Signature
```
TensorEngine* random_choice(TensorEngine *src, int n, bool replace, bool __GPU__);
```

> Example
```
TensorEngine* src = arange(1.0, 11.0, 1.0, false);

TensorEngine* picked = random_choice(src, 4, false, false);
p(picked);

free_tensor(src);
free_tensor(picked);
```

> Second Example
```
TensorEngine* src = arange(1.0, 6.0, 1.0, false);

// with replacement: values may repeat
TensorEngine* picked = random_choice(src, 8, true, true);
p(picked);

free_tensor(src);
free_tensor(picked);
```

## random_normal
```
The `random_normal` function creates a new tensor of the given shape, filled 
with values drawn from a normal (Gaussian) distribution with the specified 
`mean` and `stddev`. Values are generated using the Box-Muller transform. 
Supports both CPU and GPU tensor creation via the `__GPU__` flag.
```

> How it works
```
Each value is generated using the Box-Muller transform, which converts two 
independent uniform random numbers into one normally-distributed value:
1. Draw u1 and u2, each uniform in [0.0, 1.0) (u1 is clamped away from 0 to 
   avoid taking log(0), which would be undefined).
2. Compute magnitude = sqrt(-2 * log(u1)).
3. Compute the standard normal sample as magnitude * cos(2 * PI * u2).
This produces a value from the standard normal distribution N(0, 1) — 
centered at 0, with standard deviation 1. To shift it to the requested 
distribution N(mean, stddev), the raw standard-normal sample is scaled by 
stddev and shifted by mean: result = mean + standard_sample * stddev. This 
scale-and-shift step is why any mean/stddev combination can be produced from 
the same underlying standard-normal generator.
```

> Signature
```
TensorEngine* random_normal(int shape[], f64 mean, f64 stddev, bool __GPU__);
```

> Example
```
int shape[] = {4, 4, N};
TensorEngine* t = random_normal(shape, 0.0, 1.0, false);
p(t);

free_tensor(t);
```

> Second Example
```
int shape[] = {3, N};
TensorEngine* t = random_normal(shape, 50.0, 5.0, true);
p(t);

free_tensor(t);
```

## random_uniform
```
The `random_uniform` function creates a new tensor of the given shape, filled 
with values uniformly distributed in the half-open interval `[low, high)`. It 
is a conventionally-named wrapper around `rand_f64`, provided as the 
uniform-distribution counterpart to `random_normal`. Supports both CPU and 
GPU tensor creation.
```

> How it works
```
random_uniform simply forwards its arguments to rand_f64(low, high, shape, 
__GPU__) — it performs no independent sampling logic of its own. The two 
functions produce identical output for identical arguments; random_uniform 
exists purely so callers can pick whichever name reads more naturally 
alongside random_normal. See rand_f64's "How it works" section for the 
underlying generation mechanism.
```

> Signature
```
TensorEngine* random_uniform(int shape[], f64 low, f64 high, bool __GPU__);
```

> Example
```
int shape[] = {2, 5, N};
TensorEngine* t = random_uniform(shape, -1.0, 1.0, false);
p(t);

free_tensor(t);
```

> Second Example
```
int shape[] = {6, N};
TensorEngine* t = random_uniform(shape, 0.0, 10.0, true);
p(t);

free_tensor(t);
```

## random_binomial
```
The `random_binomial` function creates a new tensor of the given shape, 
filled with integer-valued samples drawn from a Binomial distribution with 
parameters `n_trials` (number of independent trials) and `p` (success 
probability per trial). Each output element is computed as the sum of 
`n_trials` independent Bernoulli(`p`) draws. `n_trials` must be non-negative 
and `p` must lie within `[0, 1]`, or `NULL` is returned with an error. 
Supports both CPU and GPU tensor creation.
```

> How it works
```
This uses the most direct definition of a Binomial distribution: it is the 
count of successes across n_trials independent Bernoulli(p) trials. For each 
output element, the function runs an inner loop of n_trials iterations; on 
each iteration it draws a uniform value in [0.0, 1.0) and counts it as a 
"success" if that value is less than p. The total success count across all 
n_trials iterations becomes that element's value. Because this approach 
literally simulates every individual trial, it is exact for any n_trials but 
costs O(n_trials) work per output element — for very large n_trials (in the 
millions), a normal approximation would be considerably faster, though this 
implementation favors correctness and simplicity over that optimization.
```

> Signature
```
TensorEngine* random_binomial(int shape[], int n_trials, f64 p, bool __GPU__);
```

> Example
```
int shape[] = {3, 4, N};
TensorEngine* t = random_binomial(shape, 10, 0.5, false);
p(t);

free_tensor(t);
```

> Second Example
```
int shape[] = {5, N};
TensorEngine* t = random_binomial(shape, 20, 0.3, true);
p(t);

free_tensor(t);
```

## random_poisson
```
The `random_poisson` function creates a new tensor of the given shape, 
filled with integer-valued samples drawn from a Poisson distribution with 
rate parameter `lambda`. Sampling uses Knuth's product-of-uniforms algorithm, 
which is exact for small-to-moderate `lambda` values. `lambda` must be 
non-negative, or `NULL` is returned with an error. Supports both CPU and GPU 
tensor creation.
```

> Signature
```
TensorEngine* random_poisson(int shape[], f64 lambda, bool __GPU__);
```

> Example
```
int shape[] = {3, 3, N};
TensorEngine* t = random_poisson(shape, 4.0, false);
p(t);

free_tensor(t);
```

> Second Example
```
int shape[] = {8, N};
TensorEngine* t = random_poisson(shape, 1.5, true);
p(t);

free_tensor(t);
```