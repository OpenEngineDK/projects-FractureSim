#ifndef TETRAHEDRALMESH
#define TETRAHEDRALMESH

#include <vector_types.h>
#include <cuda_runtime.h>

typedef float4 Point;
typedef int4 Tetrahedron; 
typedef uint3 Triangle;

/*
typedef float4 Point<3>;
*/
struct VertexPool {
    Point* data;
    unsigned int size;
	float4 *ABC, *Ui_t, *Ui_tminusdt, *externalForces;
	float* mass;
	int maxNumForces;
    float4* pointForces;

    VertexPool() {}
    VertexPool(unsigned int size);

    void Scale(float scale) {
        for (unsigned int i=0; i<size;i++) {
            data[i].x = data[i].x * scale;
            data[i].y = data[i].y * scale;
            data[i].z = data[i].z * scale;
        }
    }
    
    void Move(float dx, float dy, float dz) {
        for (unsigned int i=0; i<size;i++) {
            data[i].x = data[i].x + dx;
            data[i].y = data[i].y + dy;
            data[i].z = data[i].z + dz;
        }
    }
    void ConvertToCuda();
    //void ConvertToCPU() {    }
    void DeAlloc();
};

class Surface {
 public:
	Triangle* faces;
	int numFaces;

    Surface() {}
    Surface(unsigned int numTriangles) {
        numFaces = numTriangles;
        faces = (Triangle*) malloc(numFaces*sizeof(Triangle));
    }

    void ConvertToCuda();
    void DeAlloc();
};

struct ShapeFunctionDerivatives {
	float3 h1; // derivatives at node 1 w.r.t. (x,y,z)
	float3 h2; // derivatives at node 2 w.r.t. (x,y,z)
	float3 h3; // derivatives at node 3 w.r.t. (x,y,z)
	float3 h4; // derivatives at node 4 w.r.t. (x,y,z)
};

class Body {
 public:
	Tetrahedron* tetrahedra;
	int numTetrahedra;

	float* volume;
	ShapeFunctionDerivatives* shape_function_deriv;
	int4 *writeIndices;
	int numWriteIndices;

    Body() {}
    Body(unsigned int size);

    void ConvertToCuda();
    void DeAlloc();
};

struct TetrahedralTLEDState {
    // global state
	float timeStep;
	float mu, lambda; // should be per vertex or tetra

    void DeAlloc() {}
};

class Solid {
 public:
    VertexPool* vertexpool;
    Body* body;
    Surface* surface;
    TetrahedralTLEDState* state;
    Solid() {
        vertexpool = NULL;
        body = NULL;
        surface = NULL;
        state = NULL;
    }

    virtual ~Solid() {
        if (state != NULL)
            state->DeAlloc();
        if (body != NULL)
            body->DeAlloc();
        if (surface != NULL)
            surface->DeAlloc();
        if (vertexpool != NULL)
            vertexpool->DeAlloc();
        vertexpool = NULL;
        body = NULL;
        surface = NULL;
        state = NULL;
    }

    bool IsInitialized() {
        if (state == NULL ||
            body == NULL ||
            surface == NULL ||
            vertexpool == NULL)
            return false;
        else
            return true;
    }
};

#endif
