
#include "Modifier.h"
#include "Solid.h"
#include "Physics_kernels.h"
#include <Logging/Logger.h>

Modifier::Modifier(PolyShape* bVolume) : bVolume(bVolume), pIntersect(NULL),
                                         colorBuffer(NULL),
                                         vertexVboID(0), normalVboID(0), 
                                         vertexCpuPtr(NULL), normalCpuPtr(NULL),
                                         mainMemUpdated(false)
{
    CopyToGPU();
}


Modifier::~Modifier() {
}

void Modifier::Move(float x, float y, float z) {
    Matrix4f mat(make_float4(x,y,z,0));
    bVolume->Transform(&mat);
    LoadBoundingVolumeIntoVBO();  
}

void Modifier::Scale(float x, float y, float z) {
    Matrix4f mat(make_float4(x,y,z,0));
    bVolume->Transform(&mat);
    LoadBoundingVolumeIntoVBO();  
}

void Modifier::Rotate(float x, float y, float z) {
}
    
int Modifier::GetNumVertices(){
    return bVolume->numVertices;
}

void Modifier::Apply(Solid* solid) {
    if( pIntersect == NULL ) {
        CudaMemAlloc((void**)&(pIntersect), sizeof(bool)*solid->vertexpool->size);
        // Initialize points to be in front of all planes  
        CudaMemset(pIntersect, true, sizeof(bool)*solid->vertexpool->size);
    }

    // Apply modifier
    ApplyModifierStrategy(solid);
}

void Modifier::SetColorBufferForSelection(VisualBuffer* colBuf) {
    this->colorBuffer = colBuf;
}

void Modifier::Visualize() {
    if( vertexVboID > 0 ) {  
        glColor4f(0, 0, 0.8, 0.5);
        glEnable(GL_NORMALIZE); 
        glEnable(GL_COLOR_MATERIAL);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        glBindBufferARB(GL_ARRAY_BUFFER, normalVboID);
        glNormalPointer(GL_FLOAT, sizeof(float4), 0);
        glEnableClientState(GL_NORMAL_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, vertexVboID);
        glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
        glEnableClientState(GL_VERTEX_ARRAY);
  
        glDrawArrays(GL_TRIANGLES, 0, GetNumVertices());
        
        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }    
}


void Modifier::CopyToGPU() {
    bVolume->CopyToGPU();
    vertexVboID = AllocGLBuffer(sizeof(float4)*bVolume->numVertices);
    normalVboID = AllocGLBuffer(sizeof(float4)*bVolume->numNormals);
    LoadBoundingVolumeIntoVBO();
}

void Modifier::LoadBoundingVolumeIntoVBO() {
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vertexVbo, vertexVboID)); 
    loadArrayIntoVBO(bVolume->vertices, bVolume->numVertices, vertexVbo);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vertexVboID ));   

    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&normalVbo, normalVboID)); 
    loadArrayIntoVBO(bVolume->normals, bVolume->numNormals, normalVbo);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( normalVboID ));   
}


void Modifier::VisualizeNormals() {
    if( !mainMemUpdated ) {
        // Copy back all normals from VRAM
        vertexCpuPtr = (float4*)malloc(sizeof(float4)*bVolume->numVertices);
        normalCpuPtr = (float4*)malloc(sizeof(float4)*bVolume->numNormals);
        cudaError_t vStat = CudaMemcpy(vertexCpuPtr, bVolume->vertices, 
                                       sizeof(float4)*bVolume->numVertices, 
                                       cudaMemcpyDeviceToHost);

        cudaError_t nStat = CudaMemcpy(normalCpuPtr, bVolume->normals, 
                                       sizeof(float4)*bVolume->numNormals, 
                                       cudaMemcpyDeviceToHost);

        if( vStat == cudaSuccess && nStat == cudaSuccess )
            mainMemUpdated = true;
        else
            logger.info << "VisualizeNormals: Copying data device to host failed!" << logger.end;

    } else {
        glLineWidth(2.0);
        glColor4f(1,0,0,1);
        glBegin(GL_LINES);
        for( unsigned int i=0; i<bVolume->numNormals; i++ ) {
            float4 p0 = vertexCpuPtr[i];
            float4 p1 = normalCpuPtr[i];
            glVertex3f(p0.x, p0.y, p0.z);
            glVertex3f(p0.x+p1.x, p0.y+p1.y, p0.z+p1.z);
        }
        glEnd();
    }
}
