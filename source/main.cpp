
#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"
#include "../include/tensor_factory.hpp"
#include <cstdlib>

int main()
{

    // TensorEngine* tx = arange(1.0, 65.0, 1.0, false);
    // int shape6d[] = {2, 2, 2, 2, 2, 2, N};
    // tx = reshape(tx, shape6d, 6);

    // int s6d[][3] = {
    //     {0, 1, 1}, // Dim 0: index 1
    //     {0, 1, 1}, // Dim 1: index 1
    //     {1, 2, 1}, // Dim 2: index 2
    //     {N, N, N}, // Dim 3: all
    //     {N, N, N}, // Dim 4: all
    //     {N, N, N}, // Dim 5: all
    // };
    // TensorEngine* sl6d = slice(tx, s6d, 5, 0);
    // TensorEngine* ex = extract(sl6d, sl6d->size);

    // p(tx);
    // p(sl6d);
    // p(ex);

    // free_tensor(tx);
    // free_tensor(sl6d);
    // free_tensor(ex);




    TensorEngine* t = arange(1.0, 8641.0, 1.0, false);
    int shape7d[] = {4, 3, 5, 6, 2, 3, 4, N};
    t = reshape(t, shape7d, 7);

    int s7d[][3] = {
        {2, 3, 1},  // Dim 0: index 2 only
        {1, 2, 1},  // Dim 1: index 1 only
        {4, 5, 1},  // Dim 2: index 4 only
        {0, 1, 1},  // Dim 3: index 0 only
        {N, N, N},  // Dim 4: all (size 2)
        {N, N, N},  // Dim 5: all (size 3)
        {N, N, N},  // Dim 6: all (size 4)
    };
    TensorEngine* sl = slice(t, s7d, 6, 0);
    p(sl);
    TensorEngine* ex = extract(sl, sl->size);

    p(ex);

    free_tensor(t);
    free_tensor(sl);
    free_tensor(ex);




    // TensorEngine* t = arange(10.0, 0.0, -2.0, true);
    // p(t);
    // free_tensor(t);


    // TensorEngine* t = arange(1.0, 13.0, 1.0, false);
    // int shape[] = {3, 4};
    // TensorEngine* mat = reshape(t, shape, 2);

    // int s[][3] = {
    //     {0, 2, 1}, // Rows: [0, 2] step 1
    //     {1, 4, 2}  // Cols: [1, 4] step 2
    // };
    // TensorEngine* sub_mat = slice(mat, s, 2, 0);

    // p(mat);
    // p(sub_mat);

    // free_tensor(t);
    // free_tensor(mat);
    // free_tensor(sub_mat);

    // TensorEngine* t1 = ones((int[]){2, 2, 5, N}, true);
    // p(t1);
    // free_tensor(t1);


    // TensorEngine* t2 = zeros((int[]){2, 2, 5, N}, true);
    // p(t2);
    // free_tensor(t2);


    // TensorEngine* t3 = full((int[]){2, 2, 5, N}, 55.0, true);
    // p(t3);


    // TensorEngine *t4 = ones_alike(t3, false);
    // p(t4);

    
    // free_tensor(t3);
    // free_tensor(t4);

    // TensorEngine* tx = empty((int[]){3, 3, N}, false);
    // p(tx);
    // free_tensor(tx);

    return 0;
}