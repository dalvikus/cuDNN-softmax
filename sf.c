#include <stdio.h>
#include <stdlib.h> // atoi
#include <assert.h> // assert
#include "error_util.h"

const int BATCH_SIZE=1;
const int NUM_CLASSES=4;
const int NUM_ELEMENTS=BATCH_SIZE*NUM_CLASSES;
const int NUM_BYTES=sizeof(float)*NUM_ELEMENTS;

int main(int argc, const char* argv[])
{
    int class_id=0;
    if (argc > 1) {
        class_id = atoi(argv[1]);
        assert(0 <= class_id);
        assert(class_id < NUM_CLASSES);
    }
    printf("INFO: class_id=%d\n", class_id);

    // any random values
    const float host_x[NUM_CLASSES]={1.1, 2.2, 0.2, -1.7};
    // y=sf(x) in math; yi=exp(xi)/∑exp(xj)
    float host_y0[NUM_CLASSES]={0.22363628, 0.6718406 , 0.09092373, 0.01359934};
    float host_y[NUM_CLASSES]; // to be compared with host_y0

    // k: 1-based index
    // set only if k == (1 + class_id)
    // dy=[0,...,-1/yk,...0] in math; -1/yk if k=(1+class_id); otherwise, 0
    float host_dy[NUM_CLASSES]={0};
    host_dy[class_id] = -1 / host_y0[class_id];
    // dx=[y1,...,yk-1,...,yn] in math; yk-1 if k=(1+class_id); otherwise, yk
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
        CUDNN_SOFTMAX_ACCURATE,
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
        CUDNN_SOFTMAX_ACCURATE,
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
