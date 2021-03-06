#ifndef _CUDA_MEM_H_
#define _CUDA_MEM_H_

#include <Meta/CUDA.h>

cudaError_t CudaMemAlloc(void** devPtr, size_t count);
cudaError_t CudaMemset(void* devPtr, int value, size_t count);
cudaError_t CudaMemcpy( void* dst, const void* src,
                        size_t count, enum cudaMemcpyKind kind );
cudaError_t CudaFree(void* devPtr);

unsigned int AllocGLBuffer(unsigned int byteSize);
void FreeGLBuffer(unsigned int id);

void PrintAllocedMemory();

#endif
