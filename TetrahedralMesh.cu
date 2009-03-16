#include "TetrahedralMesh.h"
#include "CUDA.h"

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

Body::Body(unsigned int size) {
    CHECK_FOR_CUDA_ERROR();
    numTetrahedra = size;
    
    tetrahedra = 
        (Tetrahedron*) malloc(numTetrahedra*sizeof(Tetrahedron));

    cudaMalloc((void**)&(shape_function_deriv), 
               sizeof(ShapeFunctionDerivatives) * numTetrahedra);

    writeIndices = (int4*)malloc(sizeof(int4) * size);
    volume = (float*)malloc(sizeof(float) * size);
    CHECK_FOR_CUDA_ERROR();
}

void Body::ConvertToCuda() {
    CHECK_FOR_CUDA_ERROR();
        Tetrahedron *dTets;
        cudaMalloc((void**)&dTets, sizeof(Tetrahedron)*numTetrahedra);
        CHECK_FOR_CUDA_ERROR();
        cudaMemcpy(dTets, tetrahedra,
                   sizeof(Tetrahedron)*numTetrahedra , cudaMemcpyHostToDevice); 
        CHECK_FOR_CUDA_ERROR();
        free(tetrahedra);
        this->tetrahedra = dTets;

        float* dVolume;
        cudaMalloc((void**)&dVolume,
                   sizeof(float) * numTetrahedra);
        CHECK_FOR_CUDA_ERROR();
        cudaMemcpy(dVolume, volume,
                   sizeof(float) * numTetrahedra, cudaMemcpyHostToDevice);
        CHECK_FOR_CUDA_ERROR();
        free(volume);
        this->volume = dVolume;

        int4* dWriteIndices;
        cudaMalloc((void**)&(dWriteIndices),
                   sizeof(int4) * numWriteIndices);
        CHECK_FOR_CUDA_ERROR();
        cudaMemcpy(dWriteIndices, writeIndices, 
                   sizeof(int4) * numWriteIndices,
                   cudaMemcpyHostToDevice);
        free(writeIndices);
        writeIndices = dWriteIndices;
        CHECK_FOR_CUDA_ERROR();
}

void Body::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    cudaFree(tetrahedra);
    cudaFree(shape_function_deriv);
    cudaFree(writeIndices);
    cudaFree(volume);
    CHECK_FOR_CUDA_ERROR();
}

void Surface::ConvertToCuda() {
    CHECK_FOR_CUDA_ERROR();
    Triangle* dTriangles;
	cudaMalloc((void**)&dTriangles, sizeof(Triangle) *numFaces);
	cudaMemcpy(dTriangles, faces, 
               sizeof(Triangle) *numFaces, cudaMemcpyHostToDevice);
	free(faces);
    this->faces = dTriangles;
    CHECK_FOR_CUDA_ERROR();
}

void Surface::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    cudaFree(faces);
    CHECK_FOR_CUDA_ERROR();
}
