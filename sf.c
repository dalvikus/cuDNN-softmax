#include <stdio.h>
#include "error_util.h"

int main()
{
    const int BATCH_SIZE=1;
    const int NUM_CLASSES=4;
    const int NUM_ELEMENTS=BATCH_SIZE*NUM_CLASSES;
    const int NUM_BYTES=sizeof(float)*NUM_ELEMENTS;

    float host_x[NUM_CLASSES]={1.1, 2.2, 0.2, -1.7};
    const float c[NUM_CLASSES]={0, 0, 1, 0}; // class_id=2

    // y=sf(x) in math
    float host_y0[NUM_CLASSES]={0.22363628, 0.6718406 , 0.09092373, 0.01359934};
    float host_y[NUM_CLASSES]; // to be compared with host_y0
    // dy=[0,...,-1/yk,...0] in math
    const float host_dy[NUM_CLASSES]={0, 0, -1/0.09092373, 0};
    // dx=[y1,...,yk-1,...,yn] in math
    const float host_dx0[NUM_CLASSES]={0.22363628, 0.6718406 , 0.09092373-1, 0.01359934};
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
    float alpha = 1.f, beta = 0.f;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t fcTensor;
    checkCUDNN(cudnnCreateTensorDescriptor(&fcTensor));

    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnSetTensor4dDescriptor(fcTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1));

    checkCUDNN(cudnnSoftmaxForward(
        cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        fcTensor,
        device_x,
        &beta,
        fcTensor,
        device_y
    ));

    checkCUDNN(cudnnSoftmaxBackward(
        cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        fcTensor,
        device_y,
        fcTensor,
        device_dy,
        &beta,
        fcTensor,
        device_dx
    ));

    checkCUDNN(cudnnDestroyTensorDescriptor(fcTensor));
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
}
