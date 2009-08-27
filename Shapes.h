/**
 * Visual Shape
 *
 *
 */

#ifndef _SHAPES_H_
#define _SHAPES_H_

#include <vector>
#include <string>
#include <Meta/CUDA.h>
#include "CudaMem.h"
#include "CudaMath.h"

#define BLOCKSIZE 128

// NOTE: DO NOT CHANGE THE ORDER OF THE MEMBER VARIABLES!
// The float4 color member *MUST* be in the top due to memory layout.
struct VisualBuffer {
    VisualBuffer() : vboID(0), buf(NULL), matBuf(NULL), modelVertBuf(NULL), modelNormBuf(NULL), numElm(0), 
                     byteSize(0), numIndices(0), mode(0), enabled(true) { 
        color = make_float4(0.0,1.0,0.0,1.0); 
    }
    float4 color;
    GLuint vboID;
    float4* buf; 
    float4* matBuf;
    float4* modelVertBuf;          // Base geometry of model
    float4* modelNormBuf;
    unsigned int numElm;
    unsigned int byteSize;
    unsigned int numIndices;
    GLenum mode;
    bool enabled;
    
    void SetColor(float r, float g, float b, float alpha) {
        this->color = make_float4(r, g, b, alpha);
    }
};


struct Matrix4f {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;

    
    Matrix4f() { 
        SetIdentityMatrix();
    }

    __device__
    Matrix4f(float4 pos) {
        SetIdentityMatrixOnDevice();
        SetPos(pos.x, pos.y, pos.z);
    }

    __device__
    Matrix4f(float4 pos, float4 scale) {
        SetIdentityMatrixOnDevice();
        SetPos(pos.x, pos.y, pos.z);
        SetScale(scale.x, scale.y, scale.z);
    }

    void SetIdentityMatrix() {
        row0 = make_float4(1,0,0,0);
        row1 = make_float4(0,1,0,0);
        row2 = make_float4(0,0,1,0);
        row3 = make_float4(0,0,0,1);
    }

    __device__
    void SetIdentityMatrixOnDevice() {
        row0 = make_float4(1,0,0,0);
        row1 = make_float4(0,1,0,0);
        row2 = make_float4(0,0,1,0);
        row3 = make_float4(0,0,0,1);
    }

    __device__
    void SetPos(float x, float y, float z) { 
        row0.w = x;
        row1.w = y;
        row2.w = z;
    }

    __device__
    float4 GetPos(){
        return make_float4(row0.w,row1.w,row2.w,0);
    }

    __device__
    void SetScale(float x, float y, float z) { 
        row0.x *= x;
        row1.y *= y;
        row2.z *= z;
    }

    __device__
    void RotateY(float rot){
        row0 = make_float4(cos(rot),  0, sin(rot), 0);
        row1 = make_float4(    0,     1,     0,    0);
        row2 = make_float4(-sin(rot), 0, cos(rot), 0);      
    }

    __device__
    void CopyToBuf(float4* buf, int idx) {
        // Insert 4x4 transformation matrix into buffer        
        buf[(idx*4)+0] = row0;
        buf[(idx*4)+1] = row1;
        buf[(idx*4)+2] = row2;
        buf[(idx*4)+3] = row3;
    }


    void GetTransformationMatrix(float4* matrix) {
        matrix[0] = row0;
        matrix[1] = row1;
        matrix[2] = row2;
        matrix[3] = row3;
    }
};


struct PointShape {
    float4 point;
 
    PointShape(float4 p) : point(p) {}
    
    __device__
    void CopyToBuf(float4* buf, int idx) {
        buf[idx] = point;
    }
};


struct VectorShape {
    float4 dir;
    float4 pos;

    VectorShape(float4 dir) : dir(dir) { float4 p = {0,0,0,0}; pos = p; }
    VectorShape(float4 dir, float4 pos) : dir(dir), pos(pos) {}

    __device__
    void CopyToBuf(float4* buf, int idx) {
        buf[idx*2] = pos;
        buf[(idx*2)+1] = dir;
    }
};


struct TriangleShape {
    float4 p0,p1,p2;

    TriangleShape(float4 p0, float4 p1, float4 p2) : p0(p0), p1(p1), p2(p2) {}

    __device__
    void CopyToBuf(float4* buf, int idx) {
        buf[(idx*3)+0] = p0;
        buf[(idx*3)+1] = p1;
        buf[(idx*3)+2] = p2;
    }
};



struct PolyShape {
    float4* vertices;
    unsigned int numVertices;

    float4* normals;
    unsigned int numNormals;

    PolyShape() {}
    //PolyShape(std::string name, float scale = 1.0f);
    PolyShape(std::string name, float scaleX = 1.0f, float scaleY = 1.0f, float scaleZ = 1.0f);

    void Transform(Matrix4f* matrix) {
        float4 transMatrix[4];
        matrix->GetTransformationMatrix(transMatrix);
   
        float4* matrixPtr;
        CudaMemAlloc((void**)&(matrixPtr), sizeof(float4) * 4);
        CudaMemcpy(matrixPtr, &transMatrix[0], sizeof(float4)*4, cudaMemcpyHostToDevice);

        applyTransformation(vertices, numVertices, matrixPtr);

        CudaFree(matrixPtr);
        CHECK_FOR_CUDA_ERROR();
    }

    
    void CopyToGPU() {
        CHECK_FOR_CUDA_ERROR();
        // Copy vertices to VRAM
        float4* vertPtr;
        CudaMemAlloc((void**)&(vertPtr), sizeof(float4) * numVertices);
        CudaMemcpy(vertPtr, vertices, sizeof(float4)*numVertices, cudaMemcpyHostToDevice);
        vertices = vertPtr;
        CHECK_FOR_CUDA_ERROR();

        // Copy normals to VRAM
        float4* normPtr;
        CudaMemAlloc((void**)&(normPtr), sizeof(float4) * numNormals);
        CudaMemcpy(normPtr, normals, sizeof(float4)*numNormals, cudaMemcpyHostToDevice);
        normals = normPtr;
        CHECK_FOR_CUDA_ERROR();
    }
};



#endif //_SHAPES_H_

