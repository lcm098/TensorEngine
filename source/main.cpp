
#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"


int main()
{
    printf("===================================================\n");
    printf("   TensorEngine ND Transpose & Dot Product Tests   \n");
    printf("===================================================\n");

    int shape[] = {2, 5, 5, 2, N};

    TensorEngine* t1 = arange(1.0, 101.0, 1, true);
    t1 = reshape(t1, shape, 4);
    
    TensorEngine* t2 = arange(1.0, 101.0, 1, true);
    t2 = reshape(t2, shape, 4);

    int axes[] = {0, 1, 3, 2, N};
    TensorEngine* t2_T = T(t2, axes); // ya transpose(t2, axes)
    

    TensorEngine* t3 = dot_prod(t1, t2_T);

    p(t1);
    p(t2_T);
    p(t3);

    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t2_T);
    free_tensor(t3);


    printf("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<\n\n");
    return 0;
}