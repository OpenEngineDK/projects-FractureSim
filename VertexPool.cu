#include "VertexPool.h"

VertexPool::VertexPool() {}

VertexPool::VertexPool(unsigned int size) {
    CHECK_FOR_CUDA_ERROR();
    this->size = size;
    data = (Point*) malloc(size*sizeof(Point));

    mass = (float*)malloc(sizeof(float) * size);
    memset(mass, 0, sizeof(float) * size);

	cudaMalloc((void**)&(ABC), sizeof(float4) * size);
	cudaMalloc((void**)&(Ui_t), sizeof(float4) * size);
	cudaMalloc((void**)&(Ui_tminusdt), sizeof(float4) * size);
	cudaMalloc((void**)&(externalForces), sizeof(float4) * size);
    CHECK_FOR_CUDA_ERROR();

	cudaMemset(externalForces, 0, sizeof(float4) * size);
	cudaMemset(Ui_t, 0, sizeof(float4) * size);
	cudaMemset(Ui_tminusdt, 0, sizeof(float4) * size);
    CHECK_FOR_CUDA_ERROR();

}

void VertexPool::Scale(float scale) {
    for (unsigned int i=0; i<size;i++) {
        data[i].x = data[i].x * scale;
        data[i].y = data[i].y * scale;
        data[i].z = data[i].z * scale;
    }
}
    
void VertexPool::Move(float dx, float dy, float dz) {
    for (unsigned int i=0; i<size;i++) {
        data[i].x = data[i].x + dx;
        data[i].y = data[i].y + dy;
        data[i].z = data[i].z + dz;
    }
}

void VertexPool::Print() {
    for (unsigned int i=0; i<size; i++) {
        Point id = data[i];
        printf("v[%i] = (%f,%f,%f)\n", i, id.x, id.y, id.z);
    }
}

void VertexPool::ConvertToCuda() {
    CHECK_FOR_CUDA_ERROR();
    Point *dPoints;
    cudaMalloc((void**)&dPoints, sizeof(Point) *size);
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy(dPoints, data, sizeof(Point) *size, cudaMemcpyHostToDevice); 
    CHECK_FOR_CUDA_ERROR();
    free(data);
    this->data = dPoints;
    
    float* dMass;
    cudaMalloc((void**)&dMass, sizeof(float) * size);
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy(dMass, mass, sizeof(float) * size, cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();
    free(mass);
    mass = dMass;

	cudaMalloc((void**)&(pointForces), maxNumForces * sizeof(float4) * size);
    CHECK_FOR_CUDA_ERROR();
	cudaMemset(pointForces, 0, sizeof(float4) * maxNumForces * size);
    CHECK_FOR_CUDA_ERROR();
}

void VertexPool::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    cudaFree(data);
    cudaFree(ABC);
    cudaFree(Ui_t);
    cudaFree(Ui_tminusdt);
    cudaFree(externalForces);
    cudaFree(mass);
    cudaFree(pointForces);
    CHECK_FOR_CUDA_ERROR();
}
