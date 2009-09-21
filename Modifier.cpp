
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
    transform = new Matrix4f();
    position = make_float4(0);
    color = make_float4(1.0,0,0,1.0);
    CopyToGPU();
}


Modifier::~Modifier() {
    delete transform;
}

void Modifier::Move(float x, float y, float z) {
    float4 move = make_float4(x,y,z,0);
    position += move;
    Matrix4f mat(move);
    bVolume->Transform(&mat);

    LoadBoundingVolumeIntoVBO();  
}

void Modifier::Scale(float x, float y, float z) {
    Matrix4f mat;
    mat.SetScale(x,y,z);
    bVolume->Transform(&mat);
    LoadBoundingVolumeIntoVBO();  
}


void Modifier::RotateX(float radian) {
    Matrix4f orgin(position);
    orgin.row0.w *= -1;
    orgin.row1.w *= -1;
    orgin.row2.w *= -1;
    bVolume->Transform(&orgin);
    
    Matrix4f rot;
    rot.RotateX(radian);
    bVolume->Transform(&rot);
 
    Matrix4f rotN;
    rotN.RotateX(radian);
    bVolume->TransformNormals(&rotN);

    Matrix4f pos(position);
    bVolume->Transform(&pos);

    LoadBoundingVolumeIntoVBO();
}

void Modifier::RotateY(float radian) {
    Matrix4f orgin(position);
    orgin.row0.w *= -1;
    orgin.row1.w *= -1;
    orgin.row2.w *= -1;
    bVolume->Transform(&orgin);
    
    Matrix4f rot;
    rot.RotateY(radian);
    bVolume->Transform(&rot);

    Matrix4f rotN;
    rotN.RotateY(radian);
    bVolume->TransformNormals(&rotN);

    Matrix4f pos(position);
    bVolume->Transform(&pos);

    LoadBoundingVolumeIntoVBO();
}


void Modifier::RotateZ(float radian) {
    Matrix4f orgin(position);
    orgin.row0.w *= -1;
    orgin.row1.w *= -1;
    orgin.row2.w *= -1;
    bVolume->Transform(&orgin);
    
    Matrix4f rot;
    rot.RotateZ(radian);
    bVolume->Transform(&rot);

    Matrix4f rotN;
    rotN.RotateZ(radian);
    bVolume->TransformNormals(&rotN);

    Matrix4f pos(position);
    bVolume->Transform(&pos);

    LoadBoundingVolumeIntoVBO();
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
        glColor4f(color.x, color.y, color.z, color.w);
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


void Modifier::SelectNodes(Solid* solid) {
    // Initialize points to be in front of all planes  
    CudaMemset(pIntersect, true, sizeof(bool)*solid->vertexpool->size);
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
        mainMemUpdated = true;
    }
    else {
        cudaError_t vStat = CudaMemcpy(vertexCpuPtr, bVolume->vertices, 
                                       sizeof(float4)*bVolume->numVertices, 
                                       cudaMemcpyDeviceToHost);

        cudaError_t nStat = CudaMemcpy(normalCpuPtr, bVolume->normals, 
                                       sizeof(float4)*bVolume->numNormals, 
                                       cudaMemcpyDeviceToHost);

        //        if( vStat == cudaSuccess && nStat == cudaSuccess )
        //    mainMemUpdated = true;
        //else
        //    logger.info << "VisualizeNormals: Copying data device to host failed!" << logger.end;

        //} else {
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
