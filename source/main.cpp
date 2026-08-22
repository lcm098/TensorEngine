
#include "../include/utils.hpp"
#include "../include/arthematic.hpp"
#include "../include/utility.hpp"

int main()
{

    TensorEngine* tx = arange(1.0, 65.0, 1.0, false);
    int shape6d[] = {2, 2, 2, 2, 2, 2, N};
    tx = reshape(tx, shape6d, 6);

    int s6d[][3] = {
        {0, 1, 1}, // Dim 0: index 1
        {0, 1, 1}, // Dim 1: index 1
        {1, 2, 1}, // Dim 2: index 2
        {N, N, N}, // Dim 3: all
        {N, N, N}, // Dim 4: all
        {N, N, N}, // Dim 5: all
    };
    TensorEngine* sl6d = slice(tx, s6d);

    p(tx);
    p(sl6d);

    free_tensor(tx);
    free_tensor(sl6d);
    return 0;
}