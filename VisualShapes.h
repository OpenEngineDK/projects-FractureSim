/**
 * Visual Shape
 *
 *
 */

#ifndef _VISUAL_SHAPE_
#define _VISUAL_SHAPE_

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

enum {
    //    ELM_CENTER_OF_MASS = 0,
    STRESS_TENSORS = 0,
    NUM_BUFFERS
};

enum {
    POINTS = 1,
    LINES,
    TRIANGLES,
    POLYGONS
}; 


struct VisualBuffer {
    VisualBuffer() : vboID(0), buf(NULL), bufExt(NULL), numElm(0), 
                     byteSize(0), mode(0), enabled(true) {}
    GLuint vboID;
    float4* buf;
    float4* bufExt;
    unsigned int numElm;
    unsigned int byteSize;
    GLenum mode;
    bool enabled;
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


struct PolyShape {
    float4* vertices;
    int numVertices;

    PolyShape() {}
    PolyShape(std::string name);

    static void render();
   
    __device__
    void CopyToBuf(float4* buf, int idx) {
        // Insert 4x4 transformation matrix into buffer
        float4 row0 = {0.01,0,0,0}; // move in x direction
        float4 row1 = {0,1,0,0};
        float4 row2 = {0,0,1,0};
        float4 row3 = {0,0,0,1};
        
        
        buf[(idx*4)+0] = row0;
        
        /*buf[(idx*4)+1] = row1;
        buf[(idx*4)+2] = row2;
        buf[(idx*4)+3] = row3;
        */
    }

};




#endif _VISUAL_SHAPE_
