
#ifndef _MODIFIER_H_
#define _MODIFIER_H_

#include "Shapes.h"

class PolyShape;
class Solid;

class Modifier {
public:
    Modifier(PolyShape* bVolume);
    virtual ~Modifier();

    void Move(float x, float y, float z);
    void Scale(float x, float y, float z);
    void RotateX(float radian);
    void RotateY(float radian);
    void RotateZ(float radian);
    
    int GetNumVertices();

    void Apply(Solid* solid);
    virtual void SelectNodes(Solid* solid);    
    void SetColorBufferForSelection(VisualBuffer* colBuf);
    virtual void Visualize();
    void VisualizeNormals();

protected:
    PolyShape* bVolume;
    bool* pIntersect;
    VisualBuffer* colorBuffer;
    float4 color;

private:
    float4* vertexVbo;
    float4* normalVbo;
    GLuint vertexVboID;
    GLuint normalVboID;

    float4* vertexCpuPtr;
    float4* normalCpuPtr;

    
    float4 position;
    Matrix4f* transform;

    bool mainMemUpdated;
    
    virtual void ApplyModifierStrategy(Solid* solid) = 0;
    

    void CopyToGPU();
    void LoadBoundingVolumeIntoVBO();
    
};

#endif // _MODIFIER_H_
