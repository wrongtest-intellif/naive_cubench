#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "cudnn.h"


#define CHECK_CUDNN(f) {                               \
	cudnnStatus_t status = (f);                        \
	if (status != CUDNN_STATUS_SUCCESS) {              \
        fprintf(stderr, "Cudnn error at %d: %d\n",     \
        	__LINE__, status);                         \
        return status;                                 \
	}                                                  \
}                                                      \

#define CHECK_CUDA_RUNTIME(f) {                        \
	cudaError_t status = (f);                          \
	if (status != cudaSuccess) {                       \
        fprintf(stderr, "Cuda runtime error"           \
        	"at %d: %d\n", __LINE__, status);          \
        return status;                                 \
	}                                                  \
}                                                      \

#endif  // __CUDA_UTILS_H__
