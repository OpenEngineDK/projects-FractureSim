#include "Surface.h"

#include <Meta/CUDA.h>
#include "CudaMem.h"

Surface::Surface() {}

Surface::Surface(unsigned int numTriangles) {
    numFaces = numTriangles;
    faces = (Triangle*) malloc(numFaces*sizeof(Triangle));
}

void Surface::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    CudaFree(faces);
    CHECK_FOR_CUDA_ERROR();
}

void Surface::ConvertToCuda() {
    CHECK_FOR_CUDA_ERROR();
    Triangle* dTriangles = NULL;
	CUDA_SAFE_CALL(CudaMemAlloc( (void**)(&dTriangles), sizeof(Triangle) *numFaces));

    CudaMemcpy(dTriangles, faces, 
               sizeof(Triangle) *numFaces, cudaMemcpyHostToDevice);
	free(faces);
    this->faces = dTriangles;
    CHECK_FOR_CUDA_ERROR();
}

void Surface::Print() {
    for (unsigned int i=0; i<numFaces; i++) {
        Triangle id = faces[i];
        printf("s[%i] = (%i,%i,%i)\n", i, id.x, id.y, id.z);
    }
}
