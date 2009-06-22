#include <Meta/CUDA.h>
#include "CudaMem.h"
#include "Body.h"
#include <Logging/Logger.h>
#include <cstring>

Body::Body() {}

Body::Body(unsigned int size) {
    CHECK_FOR_CUDA_ERROR();
    numTetrahedra = size;
    tetrahedra = (Tetrahedron*) malloc(sizeof(Tetrahedron) * size);
    volume = (float*)malloc(sizeof(float) * size);
    CudaMemAlloc((void**)&(principalStress), sizeof(float4) * size);
    CudaMemAlloc((void**)&(maxStressExceeded), sizeof(bool));
    CudaMemset(maxStressExceeded, 0, 1);

    CudaMemAlloc((void**)&(shape_function_deriv), sizeof(ShapeFunctionDerivatives) * size);
    writeIndices = (int4*)malloc(sizeof(int4) * size);
    edgeSharing = (int*)malloc(sizeof(int) * 12 * size);
    for( unsigned int i=0; i<size*12; i++ ) edgeSharing[i] = -1.0f;
    
    neighbour = (int*)malloc(sizeof(int)*4*size);
    for( unsigned int i=0; i<size*4; i++ ) neighbour[i] = -1;

    crackPlaneNorm = (float4*)malloc(sizeof(float4)*size);
    for( unsigned int i=0; i<size; i++ ) crackPlaneNorm[i] = make_float4(0);

    crackPoints = (float*)malloc(sizeof(float) * 6 * size);
    for( unsigned int i=0; i<size*6; i++ ) crackPoints[i] = -1;

    CHECK_FOR_CUDA_ERROR();
}

//Body::~Body() {
//}

bool Body::IsMaxStressExceeded() {
    // Alloc buffer
    bool* exceeded = (bool*)malloc(sizeof(bool));
    // Copy bool from device to host
    CudaMemcpy(exceeded, maxStressExceeded, sizeof(bool), cudaMemcpyDeviceToHost);
    // Clear flag
    CudaMemset(maxStressExceeded, 0, 1);
    // return the flag
    return *exceeded;
}

int* Body::GetNeighbours(int tetraIdx){
    return &neighbour[tetraIdx*4];
}

void Body::GetPrincipalStress(float4* pStress) {
    CHECK_FOR_CUDA_ERROR();
    // Copy principal stress from device to host
    CudaMemcpy(pStress, principalStress, sizeof(float4) * numTetrahedra, cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();
}

void Body::GetTetrahedrons(Tetrahedron* pTetras) {
    CHECK_FOR_CUDA_ERROR();
    // Copy tetrahedrons from device to host
    CudaMemcpy(pTetras, tetrahedra, sizeof(Tetrahedron) * numTetrahedra, cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();
}


void Body::AddCrackPoint(int tetraIdx, int edgeIndex, float crackPoint) {
    crackPoints[(tetraIdx*6)+edgeIndex] = crackPoint;
}

// Add crack point to tetrahedron edge given by the two node indices.
// Returns true if tetrahedron has edge given by indices, otherwise false.
bool Body::AddCrackPoint(int tetraIdx, int nodeIdx1, int nodeIdx2, float crackPoint) {
    // Figure out which edge index in tetraIdx that corresponds to the edge given by
    // the two node indices.
    for( int edge=0; edge<6; edge++ ) {
        // Get node index 1
        int idx1 = tetrahedraMainMem[tetraIdx].GetNodeIndex(GetEdgeStartIndex(edge));
        // Get node index 2
        int idx2 = tetrahedraMainMem[tetraIdx].GetNodeIndex(GetEdgeEndIndex(edge));
                    
        //logger.info << "comparing: " << idx1 << "," << idx2 << " == " << nIdx1 << "," << nIdx2;
        // If edge is the same, tetra i and j shares an edge
        if( idx1 == nodeIdx1 && idx2 == nodeIdx2 ) {
            // Edge found, add crack point to this edge
            AddCrackPoint(tetraIdx, edge, crackPoint);       
            //logger.info << "CrackPoint added to tetra " << tetraIdx << " edge " << edge << " cp = " << crackPoint << logger.end;
            return true;
        } else if( idx1 == nodeIdx2 && idx2 == nodeIdx1 ) {
            // Edge found but neighbour is indexing in reverse order so flip crack point
            AddCrackPoint(tetraIdx, edge, 1.0f - crackPoint);       
            //logger.info << "CrackPoint added to tetra " << tetraIdx << " edge " << edge << " cp(flip) = " << 1.0f - crackPoint << logger.end;
            return true;
        }
    }
    //    logger.info << "Warning: cracked edge not found in tetrahedron - possible error in neighbour list" << logger.end;
    return false;
}

float4 Body::GetPrincipalStressNorm(int tetraIdx) {
    CHECK_FOR_CUDA_ERROR();
    float4 pStressNorm;
    // Copy tetrahedrons from device to host
    CudaMemcpy(&pStressNorm, (void**)&principalStress[tetraIdx], sizeof(float4), cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();
    return pStressNorm;
}

bool Body::HasCrackPoints(int tetraIdx) {
    return NumCrackPoints(tetraIdx) > 0;
}

int Body::NumCrackPoints(int tetraIdx) {
    int numCrackPoints = 0;
    float* cp = GetCrackPoints(tetraIdx);
    for(int i=0; i<6; i++) {
        if( cp[i] != -1 ) numCrackPoints++;
    }
    return numCrackPoints;
}

float* Body::GetCrackPoints(int tetraIdx){
    return &crackPoints[tetraIdx*6];
}



/*
void Body::CopyFromDeviceToHost() {
    // Alloc buffer
    Tetragedron* tetras = (Tetrahedron*) malloc(sizeof(Tetrahedron) * numTetrahedra);
    // Copy tetrahedrons from device to host
    CudaMemcpy(tetras, tetra, sizeof(float4) * numTetrahedra, cudaMemcpyDeviceToHost);

    // Copy principal stress from device to host
    CudaMemcpy(ps, principalStress, sizeof(float4) * numTetrahedra, cudaMemcpyDeviceToHost);
    // Max stress exceeded now point to main memory 
    maxStressExceeded = b;
    }*/


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

