#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"
#include "../include/tensor_factory.hpp"
#include "../include/gradfn.hpp"



int main() {
    
    TensorEngine* t1 = tensor((f64[]){5.0, E}, (int[]){1, N}, false);
    set_requires_grad(t1, true);

    TensorEngine* t2 = tensor((f64[]){2.0, E}, (int[]){1, N}, false);

    TensorEngine* t3 = add(t1, t2);

    p(t1);
    p(t3);

    print_graph(t3);

    backward(t3);

    printf("\n=== After Backward ===\n");

    p(t1);
    p(t2);
    p(t3);

    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);


    return 0;
}