
#include "../include/utils.hpp"
#include "../include/common.hpp"
#include "../include/arthematic.hpp"

int main()
{
    f64 array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, E};
    int shape[] = {2, 3, 2, N};

    TensorEngine *t1 = tensor(array, shape, true);
    p(t1);

    TensorEngine *t2 = tensor(array, shape, true);
    p(t2);

    TensorEngine * t3 = add(t1, t2);
    p(t3);

    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);

    return 0;
}