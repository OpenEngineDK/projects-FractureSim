#ifndef TETRAHEDRALMESH
#define TETRAHEDRALMESH

#include <vector_types.h>
#include "CUDA.h"

typedef float4 Point;
typedef int4 Tetrahedron; 
typedef uint3 Triangle;

/*
typedef float4 Point<3>;
*/
struct VertexPool {
    unsigned int size;
	unsigned int maxNumForces;

    Point* data;
	float4 *ABC, *Ui_t, *Ui_tminusdt, *externalForces, *pointForces;
	float* mass;

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

    void Print() {
        for (unsigned int i=0; i<size; i++) {
            Point id = data[i];
            printf("v[%i] = (%f,%f,%f)\n", i, id.x, id.y, id.z);
        }
    }
};

class Surface {
 public:
	Triangle* faces;
	unsigned int numFaces;

    Surface() {}
    Surface(unsigned int numTriangles) {
        numFaces = numTriangles;
        faces = (Triangle*) malloc(numFaces*sizeof(Triangle));
    }

    void ConvertToCuda();
    void DeAlloc();

    void Print() {
        for (unsigned int i=0; i<numFaces; i++) {
            Triangle id = faces[i];
            printf("s[%i] = (%i,%i,%i)\n", i, id.x, id.y, id.z);
        }
    }
};

struct ShapeFunctionDerivatives {
	float3 h1; // derivatives at node 1 w.r.t. (x,y,z)
	float3 h2; // derivatives at node 2 w.r.t. (x,y,z)
	float3 h3; // derivatives at node 3 w.r.t. (x,y,z)
	float3 h4; // derivatives at node 4 w.r.t. (x,y,z)
};

class Body {
 public:
	unsigned int numTetrahedra;
	int numWriteIndices;

	Tetrahedron* tetrahedra;
	float* volume;
	ShapeFunctionDerivatives* shape_function_deriv;
	int4 *writeIndices;

    Body() {}
    Body(unsigned int size);

    void ConvertToCuda();
    void DeAlloc();

    void Print() {
        for (unsigned int i=0; i<numTetrahedra; i++) {
            Tetrahedron id = tetrahedra[i];
            printf("b[%i] = (%i,%i,%i,%i)\n", i, id.x, id.y, id.z, id.w);
        }
    }
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

    void DeAlloc() {
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

    void Print() {
        printf("--------- vertexpool --------\n");
        vertexpool->Print();
        printf("--------- body indices --------\n");
        body->Print();
        printf("--------- surface indecies --------\n");
        surface->Print();
        printf("--------- end  --------\n");

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
