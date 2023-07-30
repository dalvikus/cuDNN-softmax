/**
    https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
        apt-get install libcudnn8-samples
    dpkg -L libcudnn8-samples
        /usr/src/cudnn_samples_v8/mnistCUDNN/error_util.h
**/

/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#ifndef __ERROR_UTIL_H__
#define __ERROR_UTIL_H__

#include <iostream>
#include <sstream>
#include <cudnn.h>

#define FatalError(s) do {                                             \
    std::cout << std::flush << "ERROR: " << s << " in " <<             \
              __FILE__ << ':' << __LINE__ << "\nAborting...\n";        \
    cudaDeviceReset();                                                 \
    exit(-1);                                                          \
} while (0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _err;                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _err << "cudnn failure (" << cudnnGetErrorString(status) << ')'; \
      FatalError(_err.str());                                          \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _err;                                            \
    if (status != 0) {                                                 \
      _err << "cuda failure (" << cudaGetErrorString(status) << ')';   \
      FatalError(_err.str());                                          \
    }                                                                  \
} while(0)

#endif  // !__ERROR_UTIL_H__
