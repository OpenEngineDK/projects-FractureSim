/**
 * Visual Shape
 *
 *
 */

#ifndef _SHAPES_H_
#define _SHAPES_H_

#include <vector>
#include <string>

#include "CUDA.h"

// NOTE: DO NOT CHANGE THE ORDER OF THE MEMBER VARIABLES!
// The float4 color member *MUST* be in the top due to memory layout.
struct VisualBuffer {
    VisualBuffer() : vboID(0), buf(NULL), matBuf(NULL), modelBuf(NULL), colorBuf(NULL), numElm(0), 
                     byteSize(0), mode(0), enabled(true) { 
        color = make_float4(0.0,1.0,0.0,1.0); 
    }
    float4 color;
    GLuint vboID;
    float4* buf; 
    float4* matBuf;
    float4* modelBuf;          // Base geometry of model
    float4* colorBuf;
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
    unsigned int numVertices;

    PolyShape() {}
    PolyShape(std::string name);

};


struct Matrix4f {
    float4 t;
    float4 s;
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;

    Matrix4f() { 
        t = make_float4(0,0,0,0);
        s = make_float4(1,1,1,0);
        
        row0 = make_float4(1,0,0,0);
        row1 = make_float4(0,1,0,0);
        row2 = make_float4(0,0,1,0);
        row3 = make_float4(0,0,0,1);
    }

    Matrix4f(float4 pos) : t(pos) {
        s = make_float4(1,1,1,0);   
    }

    __device__
    void SetPos(float x, float y, float z) { t = make_float4(x,y,z,0); }
    __device__
    void SetScale(float x, float y, float z) { s = make_float4(x,y,z,0); }

    __device__
    void CopyToBuf(float4* buf, int idx) {
        // Insert 4x4 transformation matrix into buffer        
        buf[(idx*4)+0] = make_float4( row0.x+s.x, row0.y,     row0.z,     row0.w+t.x   );
        buf[(idx*4)+1] = make_float4( row1.x,     row1.y+s.y, row1.z,     row1.w+t.y   );
        buf[(idx*4)+2] = make_float4( row2.x,     row2.y,     row2.z+s.z, row2.w+t.z   );
        buf[(idx*4)+3] = make_float4( row3.x,     row3.y,     row3.z,     row3.w       );
    }

};


#endif //_SHAPES_H_

