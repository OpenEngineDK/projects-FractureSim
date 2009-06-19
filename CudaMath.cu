
#include <Meta/CUDA.h>

#define BLOCKSIZE 128

__global__ 
void applyTransformation_k(float4* vec, unsigned int numVec, float4* matrix) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x>=numVec)
		return;
    
    int m_idx = 4*blockIdx.x;

    // Transform the model vertex
    float4 res;
    res.x = dot(matrix[m_idx + 0], vec[threadIdx.x]);
    res.y = dot(matrix[m_idx + 1], vec[threadIdx.x]);
    res.z = dot(matrix[m_idx + 2], vec[threadIdx.x]);
    res.w = dot(matrix[m_idx + 3], vec[threadIdx.x]);

    vec[me_idx] = res;
} 

void applyTransformation(float4* vec, unsigned int numVec, float4* matrix) {



    int gridSize = (int)ceil(((float)numVec)/BLOCKSIZE);
    applyTransformation_k
        <<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (vec, numVec, matrix);
    CHECK_FOR_CUDA_ERROR();
}
