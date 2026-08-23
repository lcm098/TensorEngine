#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"
#include "../include/tensor_factory.hpp"
#include "../include/gradfn.hpp"
#include "../include/initializer.hpp"
#include "../include/linear.hpp"
#include "../include/pipeline.hpp"
#include "../include/math.hpp"
#include "../include/random.hpp"
#include "../include/forward.hpp"
#include "../include/optimizer.hpp"
#include "../include/__constants__.hpp"

int main() {
    
    // TensorEngine* t1 = tensor((f64[]){5.0, E}, (int[]){1, N}, false);
    // set_requires_grad(t1, true);

    // TensorEngine* t2 = tensor((f64[]){2.0, E}, (int[]){1, N}, false);

    // TensorEngine* t3 = add(t1, t2);

    // p(t1);
    // p(t3);

    // print_graph(t3);

    // backward(t3);

    // printf("\n=== After Backward ===\n");

    // p(t1);
    // p(t2);
    // p(t3);

    // free_tensor(t1);
    // free_tensor(t2);
    // free_tensor(t3);

    

    // __pipeLine__ *p = build(
    //     Linear_Lay(2,   4,  INIT_XAVIER_NORMAL, Tanh, BIAS_NORMAL, 0.0, 0.2, true),
    //     Linear_Lay(4,  8,  INIT_XAVIER_NORMAL, Tanh, BIAS_NORMAL, 0.0, 0.2, true),
    //     Linear_Lay(8,  4,   INIT_XAVIER_NORMAL, Tanh, BIAS_NORMAL, 0.0, 0.2, true),
    //     Linear_Lay(4,  1,   INIT_XAVIER_NORMAL, Tanh, BIAS_NORMAL, 0.0, 0.2, true),
    //     NULL
    // );

    // print_pipeline(p);
    // free_pipeline(p);

    // TensorEngine *t1 = arange(1, 50, 1, false);
    // set_requires_grad(t1, true);

    // printf("ORIGINAL (x)\n");
    // p(t1);

    // printf("SIN (t2 = sin(x))\n");
    // TensorEngine *t2 = _sin(t1);
    // p(t2);

    // printf("COS (t3 = cos(x))\n");
    // TensorEngine *t3 = _cos(t1);
    // p(t3);

    // printf("BACKWARD ON SIN + COS (y = t2 + t3)\n");
    // TensorEngine *t4 = add(t2, t3);
    // backward(t4);

    // printf("\n=== Output Tensor y = sin(x) + cos(x) ===\n");
    // p(t4);

    // printf("\n=== Input Tensor x after Backward (dy/dx = cos(x) - sin(x)) ===\n");
    // p(t1);

    // free_tensor(t1);
    // free_tensor(t2);
    // free_tensor(t3);
    // free_tensor(t4);




    // random_seed(42); // optional, for reproducibility

    // int shape[] = {3, 4, N};

    // TensorEngine* u  = random_uniform(shape, -1.0, 1.0, false);
    // TensorEngine* n  = random_normal(shape, 0.0, 1.0, true);
    // TensorEngine* rn = random_randn(shape, false);
    // TensorEngine* b  = random_binomial(shape, 10, 0.3, false);
    // TensorEngine* ps = random_poisson(shape, 4.0, false);
    // TensorEngine* rf = rand_f64(0.0, 100.0, shape, true);

    // p(u); 
    // p(n); 
    // p(rn); 
    // p(b); 
    // p(ps); 
    // p(rf);

    // random_shuffle(u); // shuffles u's elements in place, flattened
    // p(u);

    // TensorEngine* picked = random_choice(u, 5, false, false); // 5 samples, no replacement
    // p(picked);

    // free_tensor(u); free_tensor(n); free_tensor(rn);
    // free_tensor(b); free_tensor(ps); free_tensor(rf);
    // free_tensor(picked);


    // random_seed(42);
    // TensorEngine *t1 = rand_f64(1, 10, (int[]){1, 2, 5, N}, false);
    // p(t1);

    // random_seed(42);
    // TensorEngine *t2 = rand_f64(1, 10, (int[]){1, 2, 5, N}, false);
    // p(t2);

    // free_tensor(t1);
    // free_tensor(t2);



    random_seed(42);

    // ------------------------------------------------------------
    // 1. Build training data: x in [-pi, pi], target = [cos(x), sin(x)]
    // ------------------------------------------------------------
    const int N_SAMPLES = 1024;

    TensorEngine *x = linspace(-PI, PI, N_SAMPLES, false);
    int xshape[] = {N_SAMPLES, 1};
    x = reshape(x, xshape, 2); // [64, 1]

    TensorEngine *cos_x = _cos(x);
    TensorEngine *sin_x = _sin(x);

    f64 *target_data = (f64*) malloc(sizeof(f64) * (N_SAMPLES * 2 + 1));
    for (int i = 0; i < N_SAMPLES; i++) {
        target_data[i*2 + 0] = cos_x->tensor[i];
        target_data[i*2 + 1] = sin_x->tensor[i];
    }
    target_data[N_SAMPLES * 2] = E;

    int target_shape[] = {N_SAMPLES, 2, N};
    TensorEngine *target = tensor(target_data, target_shape, false);
    free(target_data);
    free_tensor(cos_x);
    free_tensor(sin_x);

    // ------------------------------------------------------------
    // 2. Build a small network: 1 -> 16 -> 16 -> 2
    // ------------------------------------------------------------
    __pipeLine__ *pipe = build(
        Linear_Lay(1, 16, INIT_XAVIER_NORMAL, Tanh, BIAS_ZEROS, 0.0, 0.0, false),
        Linear_Lay(16, 16, INIT_XAVIER_NORMAL, Tanh, BIAS_ZEROS, 0.0, 0.0, false),
        Linear_Lay(16, 2, INIT_XAVIER_NORMAL, Tanh, BIAS_ZEROS, 0.0, 0.0, false),
        NULL
    );

    if (pipe == NULL) {
        fprintf(stderr, "Failed to build pipeline\n");
        return 1;
    }

    for (size_t i = 0; i < pipe->depth; i++) {
        set_requires_grad(pipe->layers[i]->weights, true);
        set_requires_grad(pipe->layers[i]->bias, true);
    }

    // Adam optimizer state, one per trainable tensor
    OptState *w_state[3];
    OptState *b_state[3];
    for (size_t i = 0; i < pipe->depth; i++) {
        w_state[i] = optstate_create(pipe->layers[i]->weights->size);
        b_state[i] = optstate_create(pipe->layers[i]->bias->size);
    }

    f64 lr = 0.01;
    int epochs = 3000;

    // ------------------------------------------------------------
    // 3. Training loop
    // ------------------------------------------------------------
    for (int epoch = 0; epoch < epochs; epoch++) {

        for (size_t i = 0; i < pipe->depth; i++) {
            zero_grad(pipe->layers[i]->weights);
            zero_grad(pipe->layers[i]->bias);
        }

        TensorEngine *pred = forward(pipe, x); // [64, 2]

        // MSE loss = mean((pred - target)^2)
        TensorEngine *diff = sub(pred, target);

        int flat_shape[] = {N_SAMPLES * 2};
        TensorEngine *flat = reshape(diff, flat_shape, 1);

        TensorEngine *sum_sq = dot_prod(flat, flat); // shape [1], sum of squares

        f64 mean_val[] = {1.0 / (N_SAMPLES * 2), E};
        int mean_shape[] = {1, N};
        TensorEngine *mean_scalar = tensor(mean_val, mean_shape, false);

        TensorEngine *loss = mlt(sum_sq, mean_scalar);

        backward(loss, NULL);

        for (size_t i = 0; i < pipe->depth; i++) {
            Layer *layer = pipe->layers[i];

            TensorEngine *new_w = adam_update(layer->weights, w_state[i], lr, 0.9, 0.999, 1e-8);
            if (new_w) { free_tensor(layer->weights); layer->weights = new_w; }

            TensorEngine *new_b = adam_update(layer->bias, b_state[i], lr, 0.9, 0.999, 1e-8);
            if (new_b) { free_tensor(layer->bias); layer->bias = new_b; }
        }

        if (epoch % 300 == 0 || epoch == epochs - 1) {
            printf("epoch %4d | loss = %f\n", epoch, loss->tensor[0]);
        }

        free_tensor(pred);
        free_tensor(diff);
        free_tensor(flat);
        free_tensor(sum_sq);
        free_tensor(mean_scalar);
        free_tensor(loss);
    }

    // ------------------------------------------------------------
    // 4. Test on a few unseen points
    // ------------------------------------------------------------
    printf("\n--- test predictions [cos(x), sin(x)] ---\n");

    f64 test_vals[] = {0.0, PI/6, PI/4, PI/2, PI, -PI/2, E};
    int test_shape[] = {6, 1, N};
    TensorEngine *test_x = tensor(test_vals, test_shape, false);

    TensorEngine *test_pred = forward(pipe, test_x);
    p(test_pred);

    printf("\n--- ground truth ---\n");
    TensorEngine *true_cos = _cos(test_x);
    TensorEngine *true_sin = _sin(test_x);
    p(true_cos);
    p(true_sin);

    // ------------------------------------------------------------
    // 5. Cleanup
    // ------------------------------------------------------------
    free_tensor(x);
    free_tensor(target);
    free_tensor(test_x);
    free_tensor(test_pred);
    free_tensor(true_cos);
    free_tensor(true_sin);

    for (size_t i = 0; i < pipe->depth; i++) {
        optstate_free(w_state[i]);
        optstate_free(b_state[i]);
    }
    free_pipeline(pipe);


    return 0;
}