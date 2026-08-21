
#include "../include/utils.hpp"
#include "../include/common.hpp"
#include "../include/arthematic.hpp"

int main()
{
    // f64 array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, E};
    // int shape[] = {2, 3, 2, N};

    // TensorEngine *t1 = tensor(array, shape, true);
    // p(t1);

    // TensorEngine *t2 = tensor(array, shape, true);
    // p(t2);

    // TensorEngine * t3 = add(t1, t2);
    // p(t3);

    // free_tensor(t1);
    // free_tensor(t2);
    // free_tensor(t3);

    // f64 arr[] = {1, E};
    // int s1[] = {1, 1, 1, 1, N};

    // TensorEngine *t4 = tensor(arr, s1, false);
    // TensorEngine *t5 = tensor(arr, s1, false);
    // p(t4);
    // p(t5);

    // TensorEngine *sum = add(t4, t5);
    // p(sum);

    // free_tensor(t4);
    // free_tensor(t5);
    // free_tensor(sum);

    // f64 arr1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, E};
    // int s2[] = {2, 3, 2, N};
    // TensorEngine* t6 = tensor(arr1, s2, true);

    // f64 arr2[] = {5, 6, E};
    // int s3[] = {2, N};

    // TensorEngine* t7 = tensor(arr2, s3, true);
    // TensorEngine* t8 = add_broad(t6, t7);

    // p(t6);
    // p(t7);
    // p(t8);

    // free_tensor(t6);
    // free_tensor(t7);
    // free_tensor(t8);


    f64 arr1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, E};
    int s2[] = {2, 3, 2, N};
    TensorEngine* t6 = tensor(arr1, s2, true);

    f64 arr2[] = {5, 6, E};
    int s3[] = {2, 1, 1, N};

    TensorEngine* t7 = tensor(arr2, s3, true);
    TensorEngine* t8 = add_broad(t6, t7);

    p(t6);
    p(t7);
    p(t8);

    free_tensor(t6);
    free_tensor(t7);
    free_tensor(t8);


    return 0;
}