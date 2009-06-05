#ifndef _VERTEX_POOL_
#define _VERTEX_POOL_

#include <Meta/CUDA.h>

class Tetrahedron;
typedef float4 Point;

struct VertexPool {
    unsigned int size;
	unsigned int maxNumForces;

    Point* data;
	float4 *ABC, *Ui_t, *Ui_tminusdt, *externalForces, *pointForces;
	float* mass;

    VertexPool();
    VertexPool(unsigned int size);
    void Scale(float scale);
    void Move(float dx, float dy, float dz);
    void Print();
    void GetTetrahedronVertices(Tetrahedron tetra, float4* vertices);
    void GetTetrahedronDisplacements(Tetrahedron tetra, float4* displacements);
    void GetTetrahedronAbsPosition(Tetrahedron tetra, float4* absPos);

    void ConvertToCuda();
    void DeAlloc();
};

#endif // _VERTEX_POOL_
