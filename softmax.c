#include <assert.h> // assert
#include <math.h> // expf
#include <stdlib.h> // malloc,free
#include <float.h> // FLT_MIN

void softmax(const float x[], float y[], int num_elements)
{
    int i;
    float* tmp=0;

    tmp = (float *) malloc(sizeof(float) * num_elements);
    assert(tmp);

    float max=FLT_MIN;
    for (i = 0; i < num_elements; ++i) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    // x' = x - max(x)
    for (i = 0; i < num_elements; ++i) {
        tmp[i] = x[i] - max;
    }

    float sum = 0.f;
    for (i = 0; i < num_elements; ++i) {
        float f = expf(tmp[i]);
        tmp[i] = f;
        sum += f;
    }

    for (i = 0; i < num_elements; ++i) {
        y[i] = tmp[i] / sum;
    }

    free(tmp);
}
