/**
 * Visual Shape
 *
 *
 */

#ifndef _SHAPES_H_
#define _SHAPES_H_

#include <OpenGL/gl.h>
#include <vector_types.h>
#include <vector>
#include <string>
#include <cufft.h>
#include <cutil.h>
#include <cuda.h>
#include <driver_types.h> // includes cudaError_t
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h> // includes cudaMalloc and cudaMemset
#include "float_utils.h"

// NOTE: DO NOT CHANGE THE ORDER OF THE MEMBER VARIABLES!
// The float4 color member *MUST* be in the top due to memory layout.
struct VisualBuffer {
    VisualBuffer() : vboID(0), buf(NULL), matBuf(NULL), numElm(0), 
                     byteSize(0), mode(0), enabled(true) { 
        color = make_float4(0.0,1.0,0.0,1.0); 
    }
    float4 color;
    GLuint vboID;
    float4* buf; 
    float4* matBuf;
    float4* modelBuf;          // Base geometry of model
    unsigned int numElm;
    unsigned int byteSize;
    unsigned int numIndices;
    GLenum mode;
    bool enabled;
    
    void SetColor(float r, float g, float b, float alpha) {
        this->color = make_float4(r, g, b, alpha);
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
    int numVertices;

    PolyShape() {}
    PolyShape(std::string name);

    
    __device__
    void CopyToBuf(float4* buf, int idx) {
        // Insert 4x4 transformation matrix into buffer
        float4 row0 = {1, 0, 0, 0}; // move in x direction
        float4 row1 = {0, 1, 0, 0};
        float4 row2 = {0, 0, 1, 0};
        float4 row3 = {0, 0, 0, 1};
        
        buf[(idx*4)+0] = row0;
        buf[(idx*4)+1] = row1;
        buf[(idx*4)+2] = row2;
        buf[(idx*4)+3] = row3;
    }

};




#endif _SHAPES_H_
