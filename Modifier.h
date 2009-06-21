
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
    void Rotate(float x, float y, float z);
    
    int GetNumVertices();

    void Apply(Solid* solid);
    void SetColorBufferForSelection(VisualBuffer* colBuf);
    virtual void Visualize();

protected:
    PolyShape* bVolume;
    bool* pIntersect;
    VisualBuffer* colorBuffer;

private:
    float4* vertexVbo;
    float4* normalVbo;
    GLuint vertexVboID;
    GLuint normalVboID;

    float4* vertexCpuPtr;
    float4* normalCpuPtr;

    bool mainMemUpdated;
    
    virtual void ApplyModifierStrategy(Solid* solid) = 0;

    void CopyToGPU();
    void LoadBoundingVolumeIntoVBO();
    void VisualizeNormals();
};

#endif // _MODIFIER_H_
