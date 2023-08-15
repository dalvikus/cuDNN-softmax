#include <stdio.h>
#include <stdlib.h> // atoi
#include <assert.h> // assert
#include "error_util.h"
#include <cstdlib> // rand
#include <ctime> // time
#include "softmax.h"

const int BATCH_SIZE = 4;
const int NUM_CLASSES = 10;
const int NUM_ELEMENTS = BATCH_SIZE*NUM_CLASSES;
const int NUM_BYTES = sizeof(float) * NUM_ELEMENTS;

void set_x(float x[], int num_elements)
{
    for (int i = 0; i < num_elements; ++i) {
        float f = std::rand(); // [0,RAND_MAX]
        f /= RAND_MAX; // [0,1]
        f -= 0.5f; // [-0.5,0.5]
        f *= 2; // [-1,1]
        x[i] = f;
    }
}

int main(void)
{
    std::srand(std::time(nullptr));
    int class_ids[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; ++i) {
        int class_id = std::rand() % NUM_CLASSES;
        class_ids[i] = class_id;
        printf("class_ids[%d]: %d\n", i, class_id);
    }

    float host_y0[NUM_ELEMENTS];
    float host_x[NUM_ELEMENTS];
    set_x(host_x, NUM_ELEMENTS);
    for (int k = 0; k < BATCH_SIZE; ++k) {
        const float *x_ptr = host_x + k * NUM_CLASSES; // host_x[k][...]
        float *y_ptr = host_y0 + k * NUM_CLASSES; // host_y0[k][...]
        printf("%p\n", x_ptr);
        softmax(x_ptr, y_ptr, NUM_CLASSES);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            printf("[%d][%d]: %f -> %f\n", k, i, x_ptr[i], y_ptr[i]);
        }
    }
    float host_y[NUM_ELEMENTS]; // to be compared with host_y0

/**
note that
    i=class_id: fixed

from the definition of softmax
    yj = sf(xj) = exp(xj) / ∑exp(xk)    (1)
                            k

cross-entropy loss:
    L = - ∑ ck * ln(yk) (2)
          k
where
    ck = 1 if k=i
       = 0 otherwise

dL/dy
-----
from (2)
    dL/dyj = -1/yi if j=i       (3)
           = 0     otherwise

dL/dx
-----
dL/dxj = ∑ dL/dyk * dyk/dxj
         k

from (3)
    dL/dxj = -1/yi * dyi/dxj    (4)

note that, from (1)
    ln(yj) = xj - ln(∑exp(xk))  (5)
                     k

directly
    dln(yj)/dxk = 1/yj * dyj/dxk
note that this is a matrix.

compared to (4), for i-th row in dln(yj)/dxk

    dL/dxj = -dln(yi)/dxk   (6)

from (1)
    dln(yi)/dxk = d(xi - ln(∑exp(xj)))/dxk
                            j
                = dxi/dxk - dln(∑exp(xj))/dxk
                                j
                = dxi/dxk - (d(∑exp(xj))/dxk) / ∑exp(xj)
                               j                j
                = dxi/dxk - exp(xk)/∑exp(xj)
                                    j
                = dxi/dxk - yk
inserting it into (6), we have
    dL/dxj = yk - dxi/dxk   (7)

in other words,
    dL/dxk = yk - 1 if k=i      (8)
           = yk     otherwise
**/
    float host_dy[NUM_ELEMENTS]={0};
    // from (3)
    for (int k = 0; k < BATCH_SIZE; ++k) {
        int class_id = class_ids[k];
        float *y_ptr = host_dy + k * NUM_CLASSES; // host_dy[k][...]
        float *y0_ptr = host_y0 + k * NUM_CLASSES; // host_y0[k][...]
        y_ptr[class_id] = -1 / y0_ptr[class_id];
    }
    // from (8)
    float host_dx0[NUM_ELEMENTS];
    for (int k = 0; k < BATCH_SIZE; ++k) {
        int class_id = class_ids[k];
        float *dx0_ptr = host_dx0 + k * NUM_CLASSES; // host_dx0[k][...]
        float *y0_ptr = host_y0 + k * NUM_CLASSES; // host_y0[k][...]
        for (int i = 0; i < NUM_CLASSES; ++i) {
            dx0_ptr[i] = y0_ptr[i];
            if (i == class_id)
                dx0_ptr[i] -= 1;
        }
    }
    float host_dx[NUM_ELEMENTS]; // to be compared with host_dx0

    float *device_x; // copied from host_x
    float *device_y; // copied to host_y
    float *device_dx; // copied to host_dx
    float *device_dy; // copied from host_dy
    checkCudaErrors(cudaMalloc(&device_x, NUM_BYTES));
    checkCudaErrors(cudaMalloc(&device_y, NUM_BYTES));
    checkCudaErrors(cudaMalloc(&device_dx, NUM_BYTES));
    checkCudaErrors(cudaMalloc(&device_dy, NUM_BYTES));

    checkCudaErrors(cudaMemcpy(device_x, host_x, NUM_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_dy, host_dy, NUM_BYTES, cudaMemcpyHostToDevice));

    // cuDNN...
    const float alpha = 1.f, beta = 0.f;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t sfTensor;
    checkCUDNN(cudnnCreateTensorDescriptor(&sfTensor));

    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnSetTensor4dDescriptor(sfTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1));

    checkCUDNN(cudnnSoftmaxForward(
        cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE, // x' = x - max(x)
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        sfTensor,
        device_x,
        &beta,
        sfTensor,
        device_y
    ));

    checkCUDNN(cudnnSoftmaxBackward(
        cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE, // x' = x - max(x)
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        sfTensor,
        device_y,
        sfTensor,
        device_dy,
        &beta,
        sfTensor,
        device_dx
    ));

    checkCUDNN(cudnnDestroyTensorDescriptor(sfTensor));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    checkCudaErrors(cudaMemcpy(host_y, device_y, NUM_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_dx, device_dx, NUM_BYTES, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_x));
    checkCudaErrors(cudaFree(device_y));
    checkCudaErrors(cudaFree(device_dx));
    checkCudaErrors(cudaFree(device_dy));

    for (int k = 0; k < BATCH_SIZE; ++k) {
        int class_id = class_ids[k];
        printf("samples[%d]: class_id=%d\n", k, class_id);
        float *y0_ptr = host_y0 + k * NUM_CLASSES; // host_y0[k][...]
        float *y_ptr = host_y + k * NUM_CLASSES; // host_y[k][...]
        float *dx0_ptr = host_dx0 + k * NUM_CLASSES; // host_dx0[k][...]
        float *dx_ptr = host_dx + k * NUM_CLASSES; // host_dx[k][...]
        for (int i = 0; i < NUM_CLASSES; ++i) {
            printf("class[%d]: ...\n", i);
            float y0 = y0_ptr[i];
            float y = y_ptr[i];
            printf("y: %f, %f; %f%%\n", y0,y, (y0-y)/y0*100);
            float dx0 = dx0_ptr[i];
            float dx = dx_ptr[i];
            printf("dx: %f, %f; %f%%\n", dx0,dx, (dx0-dx)/dx0*100);
        }
    }

    return 0;
}
