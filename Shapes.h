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
        s = make_float4(0,0,0,0);   
    }

    __device__
    void SetPos(float x, float y, float z) { t = make_float4(x,y,z,0); }
    __device__
    void SetScale(float x, float y, float z) { s = make_float4(x,y,z,0); }

    __device__
    void CopyToBuf(float4* buf, int idx) {
        // Insert 4x4 transformation matrix into buffer        
        buf[(idx*4)+0] = make_float4( row0.x*s.x, row0.y,     row0.z,     row0.w+t.x   );
        buf[(idx*4)+1] = make_float4( row1.x,     row1.y*s.y, row1.z,     row1.w+t.y   );
        buf[(idx*4)+2] = make_float4( row2.x,     row2.y,     row2.z*s.z, row2.w+t.z   );
        buf[(idx*4)+3] = make_float4( row3.x,     row3.y,     row3.z,     row3.w       );
    }

    /*
    void  (double x1, double y1, double z1, double x2, double y2, double z2)
{
   // Build a transform as if you were at a point (x1,y1,z1), and
     // looking at a point (x2,y2,z2) 

   double ViewOut[3];      // the View or "new Z" vector
   double ViewUp[3];       // the Up or "new Y" vector
   double ViewRight[3];    // the Right or "new X" vector

   double ViewMagnitude;   // for normalizing the View vector
   double UpMagnitude;     // for normalizing the Up vector
   double UpProjection;    // magnitude of projection of View Vector on World UP

   // first, calculate and normalize the view vector
   ViewOut[0] = x2-x1;
   ViewOut[1] = y2-y1;
   ViewOut[2] = z2-z1;
   ViewMagnitude = sqrt(ViewOut[0]*ViewOut[0] + ViewOut[1]*ViewOut[1]+
      ViewOut[2]*ViewOut[2]);

   // invalid points (not far enough apart)
   if (ViewMagnitude < .000001)
      return (-1);

   // normalize. This is the unit vector in the "new Z" direction
   ViewOut[0] = ViewOut[0]/ViewMagnitude;
   ViewOut[1] = ViewOut[1]/ViewMagnitude;
   ViewOut[2] = ViewOut[2]/ViewMagnitude;

   // Now the hard part: The ViewUp or "new Y" vector

   // dot product of ViewOut vector and World Up vector gives projection of
   // of ViewOut on WorldUp
   UpProjection = ViewOut[0]*WorldUp[0] + ViewOut[1]*WorldUp[1]+
   ViewOut[2]*WorldUp[2];

   // first try at making a View Up vector: use World Up
   ViewUp[0] = WorldUp[0] - UpProjection*ViewOut[0];
   ViewUp[1] = WorldUp[1] - UpProjection*ViewOut[1];
   ViewUp[2] = WorldUp[2] - UpProjection*ViewOut[2];

   // Check for validity:
   UpMagnitude = ViewUp[0]*ViewUp[0] + ViewUp[1]*ViewUp[1] + ViewUp[2]*ViewUp[2];

   if (UpMagnitude < .0000001)
   {
      //Second try at making a View Up vector: Use Y axis default  (0,1,0)
      ViewUp[0] = -ViewOut[1]*ViewOut[0];
      ViewUp[1] = 1-ViewOut[1]*ViewOut[1];
      ViewUp[2] = -ViewOut[1]*ViewOut[2];

      // Check for validity:
      UpMagnitude = ViewUp[0]*ViewUp[0] + ViewUp[1]*ViewUp[1] + ViewUp[2]*ViewUp[2];

      if (UpMagnitude < .0000001)
      {
          //Final try at making a View Up vector: Use Z axis default  (0,0,1)
          ViewUp[0] = -ViewOut[2]*ViewOut[0];
          ViewUp[1] = -ViewOut[2]*ViewOut[1];
          ViewUp[2] = 1-ViewOut[2]*ViewOut[2];

          // Check for validity:
          UpMagnitude = ViewUp[0]*ViewUp[0] + ViewUp[1]*ViewUp[1] + ViewUp[2]*ViewUp[2];

          if (UpMagnitude < .0000001)
              return(-1);
      }
   }

   // normalize the Up Vector
   UpMagnitude = sqrt(UpMagnitude);
   ViewUp[0] = ViewUp[0]/UpMagnitude;
   ViewUp[1] = ViewUp[1]/UpMagnitude;
   ViewUp[2] = ViewUp[2]/UpMagnitude;

   // Calculate the Right Vector. Use cross product of Out and Up.
   ViewRight[0] = -ViewOut[1]*ViewUp[2] + ViewOut[2]*ViewUp[1];
   ViewRight[1] = -ViewOut[2]*ViewUp[0] + ViewOut[0]*ViewUp[2];
   ViewRight[2] = -ViewOut[0]*ViewUp[1] + ViewOut[1]*ViewUp[0];

   // Plug values into rotation matrix R
   ViewRotationMatrix[0]=ViewRight[0];
   ViewRotationMatrix[1]=ViewRight[1];
   ViewRotationMatrix[2]=ViewRight[2];
   ViewRotationMatrix[3]=0;

   ViewRotationMatrix[4]=ViewUp[0];
   ViewRotationMatrix[5]=ViewUp[1];
   ViewRotationMatrix[6]=ViewUp[2];
   ViewRotationMatrix[7]=0;

   ViewRotationMatrix[8]=ViewOut[0];
   ViewRotationMatrix[9]=ViewOut[1];
   ViewRotationMatrix[10]=ViewOut[2];
   ViewRotationMatrix[11]=0;

   // Plug values into translation matrix T
   MoveFill(ViewMoveMatrix,-x1,-y1,-z1);

   // build the World Transform
   MatrixMultiply(ViewRotationMatrix,ViewMoveMatrix,WorldTransform);

   return(0);
}
    */
};


#endif //_SHAPES_H_

