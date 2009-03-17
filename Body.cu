#include "CUDA.h"

#include "Body.h"


Body::Body() {}

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

void Body::Print() {
    for (unsigned int i=0; i<numTetrahedra; i++) {
        Tetrahedron id = tetrahedra[i];
        printf("b[%i] = (%i,%i,%i,%i)\n", i, id.x, id.y, id.z, id.w);
    }
}

