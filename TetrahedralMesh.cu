#include "TetrahedralMesh.h"
#include "CUDA.h"

VertexPool::VertexPool(unsigned int size) {
    this->size = size;
    data = (Point*) malloc(size*sizeof(Point));

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

void VertexPool::ConvertToCuda() {
    Point *dPoints; 
    cudaMalloc((void**)&dPoints, sizeof(Point) *size);
    cudaMemcpy(dPoints, data, sizeof(Point) *size, cudaMemcpyHostToDevice); 
    free(data);
    this->data = dPoints;
    CHECK_FOR_CUDA_ERROR();
}

void VertexPool::DeAlloc() {
    cudaFree(data);
    cudaFree(ABC);
    cudaFree(Ui_t);
    cudaFree(Ui_tminusdt);
    cudaFree(externalForces);
    cudaFree(mass);
    cudaFree(pointForces);
}

Body::Body(unsigned int size) {
    numTetrahedra = size;
    
    tetrahedra = 
        (Tetrahedron*) malloc(numTetrahedra*sizeof(Tetrahedron));

    cudaMalloc((void**)&(shape_function_deriv), 
               sizeof(ShapeFunctionDerivatives) * numTetrahedra);
}

void Body::ConvertToCuda() {
        Tetrahedron *dTets;
        cudaMalloc((void**)&dTets, sizeof(Tetrahedron)*numTetrahedra);
        cudaMemcpy(dTets, tetrahedra,
                   sizeof(Tetrahedron)*numTetrahedra , cudaMemcpyHostToDevice); 
        free(tetrahedra);
        this->tetrahedra = dTets;
}

void Body::DeAlloc() {
    cudaFree(tetrahedra);
    cudaFree(shape_function_deriv);
    cudaFree(writeIndices);
    cudaFree(volume);
}

void Surface::ConvertToCuda() {
    Triangle* dTriangles;
	cudaMalloc((void**)&dTriangles, sizeof(Triangle) *numFaces);
	cudaMemcpy(dTriangles, faces, 
               sizeof(Triangle) *numFaces, cudaMemcpyHostToDevice);
	free(faces);
    this->faces = dTriangles;
}

void Surface::DeAlloc() {
    cudaFree(faces);
}
