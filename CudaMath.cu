
#include <Meta/CUDA.h>

#define BLOCKSIZE 128

__global__ 
void applyTransformation_k(float4* vec, unsigned int numVec, float4* matrix) {


} 

void applyTransformation(float4* vec, unsigned int numVec, float4* matrix) {
    int gridSize = (int)ceil(((float)numVec)/BLOCKSIZE);
    applyTransformation_k
        <<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (vec, numVec, matrix);
    CHECK_FOR_CUDA_ERROR();
}
