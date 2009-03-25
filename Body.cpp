#include "CUDA.h"
#include "CudaMem.h"
#include "Body.h"


Body::Body() {}

Body::Body(unsigned int size) {
    CHECK_FOR_CUDA_ERROR();
    numTetrahedra = size;
    
    tetrahedra = 
        (Tetrahedron*) malloc(numTetrahedra*sizeof(Tetrahedron));

    CudaMemAlloc((void**)&(shape_function_deriv), 
               sizeof(ShapeFunctionDerivatives) * numTetrahedra);

    writeIndices = (int4*)malloc(sizeof(int4) * size);
    volume = (float*)malloc(sizeof(float) * size);
    CHECK_FOR_CUDA_ERROR();
}

void Body::ConvertToCuda() {
    CHECK_FOR_CUDA_ERROR();
        Tetrahedron *dTets;
        CudaMemAlloc((void**)&dTets, sizeof(Tetrahedron)*numTetrahedra);
        CHECK_FOR_CUDA_ERROR();
        CudaMemcpy(dTets, tetrahedra,
                   sizeof(Tetrahedron)*numTetrahedra , cudaMemcpyHostToDevice); 
        CHECK_FOR_CUDA_ERROR();
        free(tetrahedra);
        this->tetrahedra = dTets;

        float* dVolume;
        CudaMemAlloc((void**)&dVolume,
                   sizeof(float) * numTetrahedra);
        CHECK_FOR_CUDA_ERROR();
        CudaMemcpy(dVolume, volume,
                   sizeof(float) * numTetrahedra, cudaMemcpyHostToDevice);
        CHECK_FOR_CUDA_ERROR();
        free(volume);
        this->volume = dVolume;

        int4* dWriteIndices;
        CudaMemAlloc((void**)&(dWriteIndices),
                   sizeof(int4) * numWriteIndices);
        CHECK_FOR_CUDA_ERROR();
        CudaMemcpy(dWriteIndices, writeIndices, 
                   sizeof(int4) * numWriteIndices,
                   cudaMemcpyHostToDevice);
        free(writeIndices);
        writeIndices = dWriteIndices;
        CHECK_FOR_CUDA_ERROR();
}

void Body::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    CudaFree(tetrahedra);
    CudaFree(shape_function_deriv);
    CudaFree(writeIndices);
    CudaFree(volume);
    CHECK_FOR_CUDA_ERROR();
}

void Body::Print() {
    for (unsigned int i=0; i<numTetrahedra; i++) {
        Tetrahedron id = tetrahedra[i];
        printf("b[%i] = (%i,%i,%i,%i)\n", i, id.x, id.y, id.z, id.w);
    }
}

