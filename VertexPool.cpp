#include "VertexPool.h"
#include "CudaMem.h"
#include "Body.h"
#include <cstring>

VertexPool::VertexPool() {}

VertexPool::VertexPool(unsigned int size) {
    CHECK_FOR_CUDA_ERROR();
    this->size = size;
    data = (Point*) malloc(size*sizeof(Point));

    mass = (float*)malloc(sizeof(float) * size);
    memset(mass, 0, sizeof(float) * size);

	CudaMemAlloc((void**)&(ABC), sizeof(float4) * size);
	CudaMemAlloc((void**)&(Ui_t), sizeof(float4) * size);
	CudaMemAlloc((void**)&(Ui_tminusdt), sizeof(float4) * size);
	CudaMemAlloc((void**)&(externalForces), sizeof(float4) * size);
    CHECK_FOR_CUDA_ERROR();

	CudaMemset(externalForces, 0, sizeof(float4) * size);
	CudaMemset(Ui_t, 0, sizeof(float4) * size);
	CudaMemset(Ui_tminusdt, 0, sizeof(float4) * size);
    CHECK_FOR_CUDA_ERROR();

}

void VertexPool::Scale(float scale) {
    Scale(scale, scale, scale);
}


void VertexPool::Scale(float x, float y, float z) {
    for (unsigned int i=0; i<size;i++) {
        data[i].x = data[i].x * x;
        data[i].y = data[i].y * y;
        data[i].z = data[i].z * z;
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

void VertexPool::GetTetrahedronVertices(Tetrahedron tetra, float4* vertices) {
    CHECK_FOR_CUDA_ERROR();
    // Copy tetrahedrons vertices from device to host
    CudaMemcpy(&vertices[0], (void**)&data[tetra.x], sizeof(float4), cudaMemcpyDeviceToHost);    
    CudaMemcpy(&vertices[1], (void**)&data[tetra.y], sizeof(float4), cudaMemcpyDeviceToHost);    
    CudaMemcpy(&vertices[2], (void**)&data[tetra.z], sizeof(float4), cudaMemcpyDeviceToHost);    
    CudaMemcpy(&vertices[3], (void**)&data[tetra.w], sizeof(float4), cudaMemcpyDeviceToHost);      
    CHECK_FOR_CUDA_ERROR();
}

void VertexPool::GetTetrahedronDisplacements(Tetrahedron tetra, float4* displacements) {
    CHECK_FOR_CUDA_ERROR();
    CudaMemcpy(&displacements[0], (void**)&Ui_t[tetra.x], sizeof(float4), cudaMemcpyDeviceToHost);    
    CudaMemcpy(&displacements[1], (void**)&Ui_t[tetra.y], sizeof(float4), cudaMemcpyDeviceToHost);    
    CudaMemcpy(&displacements[2], (void**)&Ui_t[tetra.z], sizeof(float4), cudaMemcpyDeviceToHost);    
    CudaMemcpy(&displacements[3], (void**)&Ui_t[tetra.w], sizeof(float4), cudaMemcpyDeviceToHost);      
    CHECK_FOR_CUDA_ERROR();    
}

void VertexPool::GetTetrahedronAbsPosition(Tetrahedron tetra, float4* absPos) {
    // Get init pos
    GetTetrahedronVertices(tetra, absPos);
    // Alloc for displacements
    float4* disp = (float4*)malloc(sizeof(float4) * 4);
    GetTetrahedronDisplacements(tetra, disp);
    // Add the two together to get absolute position
    for(int i=0; i<4; i++)
        absPos[i] += disp[i];
}


void VertexPool::ConvertToCuda() {
    CHECK_FOR_CUDA_ERROR();
    Point *dPoints;
    CudaMemAlloc((void**)&dPoints, sizeof(Point) *size);
    CHECK_FOR_CUDA_ERROR();
    CudaMemcpy(dPoints, data, sizeof(Point) *size, cudaMemcpyHostToDevice); 
    CHECK_FOR_CUDA_ERROR();
    free(data);
    this->data = dPoints;
    
    float* dMass;
    CudaMemAlloc((void**)&dMass, sizeof(float) * size);
    CHECK_FOR_CUDA_ERROR();
    CudaMemcpy(dMass, mass, sizeof(float) * size, cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();
    free(mass);
    mass = dMass;

	CudaMemAlloc((void**)&(pointForces), maxNumForces * sizeof(float4) * size);
    CHECK_FOR_CUDA_ERROR();
	CudaMemset(pointForces, 0, sizeof(float4) * maxNumForces * size);
    CHECK_FOR_CUDA_ERROR();
}

void VertexPool::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    CudaFree(data);
    CudaFree(ABC);
    CudaFree(Ui_t);
    CudaFree(Ui_tminusdt);
    CudaFree(externalForces);
    CudaFree(mass);
    CudaFree(pointForces);
    CHECK_FOR_CUDA_ERROR();
}
