#include <stdio.h>
#include <stdlib.h> // atoi
#include <assert.h> // assert
#include "error_util.h"
#include <cstdlib> // rand
#include <ctime> // time
#include "softmax.h"

const int BATCH_SIZE=1;
const int NUM_CLASSES=4;
const int NUM_ELEMENTS=BATCH_SIZE*NUM_CLASSES;
const int NUM_BYTES=sizeof(float)*NUM_ELEMENTS;

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
    int class_id=std::rand() % NUM_CLASSES;
    printf("INFO: class_id=%d\n", class_id);

    float host_y0[NUM_CLASSES];
    float host_x[NUM_CLASSES];
    set_x(host_x, NUM_CLASSES);
    softmax(host_x, host_y0, NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("[%d]: %f -> %f\n", i, host_x[i], host_y0[i]);
    }
    float host_y[NUM_CLASSES]; // to be compared with host_y0

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
    float host_dy[NUM_CLASSES]={0};
    // from (3)
    host_dy[class_id] = -1 / host_y0[class_id];
    // from (8)
    float host_dx0[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; ++i) {
        host_dx0[i] = host_y0[i];
        if (i == class_id)
            host_dx0[i] -= 1;
    }
    float host_dx[NUM_CLASSES]; // to be compared with host_dx0

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
        CUDNN_SOFTMAX_ACCURATE, // x'=x-max(x)
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
        CUDNN_SOFTMAX_ACCURATE, // x'=x-max(x)
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

    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("[%d]: ...\n", i);
        float y0=host_y0[i];
        float y=host_y[i];
        printf("y: %f, %f; %f%%\n", y0,y, (y0-y)/y0*100);
        float dx0=host_dx0[i];
        float dx=host_dx[i];
        printf("dx: %f, %f; %f%%\n", dx0,dx, (dx0-dx)/dx0*100);
    }

    return 0;
}
