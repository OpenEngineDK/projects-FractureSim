#include "Surface.h"
#include "CUDA.h"

Surface::Surface() {}

Surface::Surface(unsigned int numTriangles) {
    numFaces = numTriangles;
    faces = (Triangle*) malloc(numFaces*sizeof(Triangle));
}

void Surface::DeAlloc() {
    CHECK_FOR_CUDA_ERROR();
    cudaFree(faces);
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

void Surface::Print() {
    for (unsigned int i=0; i<numFaces; i++) {
        Triangle id = faces[i];
        printf("s[%i] = (%i,%i,%i)\n", i, id.x, id.y, id.z);
    }
}
