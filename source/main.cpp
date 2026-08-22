
#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"

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
    // TensorEngine* sl6d = slice(tx, s6d);

    // p(tx);
    // p(sl6d);

    // free_tensor(tx);
    // free_tensor(sl6d);


    // TensorEngine* t = arange(10.0, 0.0, -2.0, true);
    // p(t);
    // free_tensor(t);


    TensorEngine* t = arange(1.0, 13.0, 1.0, false);
    int shape[] = {3, 4};
    TensorEngine* mat = reshape(t, shape, 2);

    int s[][3] = {
        {0, 2, 1}, // Rows: [0, 2] step 1
        {1, 4, 2}  // Cols: [1, 4] step 2
    };
    TensorEngine* sub_mat = slice(mat, s, 2, 0);

    p(mat);
    p(sub_mat);

    free_tensor(t);
    free_tensor(mat);
    free_tensor(sub_mat);


    return 0;
}