#include "CudaMem.h"

#include <map>

static size_t alloced = 0;
static std::map<void*, unsigned int> memMap;
static size_t glAlloced = 0;
static std::map<unsigned int, unsigned int> glMemMap;

cudaError_t CudaMemAlloc(void** devPtr, size_t count) {
    //printf("cuda alloced memory: %lu bytes\n", count);

    alloced += count;
    cudaError_t errorCode = cudaMalloc(devPtr, count);
    memMap[*devPtr] = count;
    return errorCode;
}

cudaError_t CudaMemset(void* devPtr, int value, size_t count) {
    return cudaMemset(devPtr, value, count);
}

cudaError_t CudaMemcpy( void* dst, const void* src, size_t count, enum cudaMemcpyKind kind ) {
    return cudaMemcpy(dst, src, count, kind);
} 

cudaError_t CudaFree(void* devPtr) {
    std::map<void*, unsigned int>::iterator iter = memMap.find(devPtr);
    if (iter == memMap.end()) {
        printf("dealloc of unalloced memory, with pointer: %lu\n",
               (unsigned long) devPtr);
        exit(-1);
    }
    alloced -= memMap[devPtr];
    return cudaFree(devPtr);
}

unsigned int AllocGLBuffer(unsigned int byteSize) {
    CHECK_FOR_GL_ERROR();
    CHECK_FOR_CUDA_ERROR();

    // create buffer object
    GLuint vboID = 0;
    glGenBuffers( 1, &vboID);
    CHECK_FOR_GL_ERROR();

    // Bind buffer
    glBindBuffer( GL_ARRAY_BUFFER, vboID);
    CHECK_FOR_GL_ERROR();

    // initialize buffer object
    glBufferData( GL_ARRAY_BUFFER, byteSize, NULL, GL_DYNAMIC_DRAW);
    CHECK_FOR_GL_ERROR(); //check for memory alloc

    // Unbind buffer
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    CHECK_FOR_GL_ERROR();

    // Register buffer object with CUDA
    cudaGLRegisterBufferObject(vboID);

    // if you get an: "CUDA Error: invalid argument" from this
    // line you properbly ran out of memory
    CHECK_FOR_CUDA_ERROR();


    //printf("glGenBufferID: %u\n", vboID);
    //printf("gl alloced memory: %u bytes\n", byteSize);

    glAlloced += byteSize;
    glMemMap[vboID] = byteSize;
    return vboID;
}

void FreeGLBuffer(unsigned int id) {
    std::map<unsigned int, unsigned int>::iterator iter = glMemMap.find(id);
    if (iter == glMemMap.end()) {
        printf("error in dealloc of gl unalloced memory, with id %u\n", id);
        exit(-1);
    }
    glAlloced -= glMemMap[id];
    glDeleteBuffers(1, (GLuint*)&id);
    CHECK_FOR_GL_ERROR();
}

void PrintAllocedMemory() {
    //printf("cuda alloced memory: %lu bytes\n", alloced);
    //printf("cuda alloced memory: %f kb\n", alloced/1024.0f);
    printf("cuda alloced memory: %f Mb\n", alloced/1024.0f/1024.0f);
    //printf("gl alloced memory: %lu bytes\n", glAlloced);
    //printf("gl alloced memory: %f kb\n", glAlloced/1024.0f);
    printf("gl alloced memory: %f Mb\n", glAlloced/1024.0f/1024.0f);
}
