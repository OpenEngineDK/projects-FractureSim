
#include "CudaMem.h"


cudaError_t CudaMemAlloc(void** devPtr, size_t count) {
    return cudaMalloc(devPtr, count);
}

cudaError_t CudaMemset(void* devPtr, int value, size_t count) {
    return cudaMemset(devPtr, value, count);
}

cudaError_t CudaMemcpy( void* dst, const void* src, size_t count, enum cudaMemcpyKind kind ) {
    return cudaMemcpy(dst, src, count, kind);
} 

cudaError_t CudaFree(void* devPtr) {
    return cudaFree(devPtr);
}
