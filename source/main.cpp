#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"
#include "../include/tensor_factory.hpp"
#include "../include/gradfn.hpp"
#include "../include/initializer.hpp"
#include "../include/linear.hpp"
#include "../include/pipeline.hpp"
#include "../include/math.hpp"
#include <cstdio>

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

    TensorEngine *t1 = arange(1, 50, 1, false);
    set_requires_grad(t1, true);

    printf("ORIGINAL (x)\n");
    p(t1);

    printf("SIN (t2 = sin(x))\n");
    TensorEngine *t2 = _sin(t1);
    p(t2);

    printf("COS (t3 = cos(x))\n");
    TensorEngine *t3 = _cos(t1);
    p(t3);

    printf("BACKWARD ON SIN + COS (y = t2 + t3)\n");
    TensorEngine *t4 = add(t2, t3);
    backward(t4);

    printf("\n=== Output Tensor y = sin(x) + cos(x) ===\n");
    p(t4);

    printf("\n=== Input Tensor x after Backward (dy/dx = cos(x) - sin(x)) ===\n");
    p(t1);

    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);
    free_tensor(t4);

    return 0;
}