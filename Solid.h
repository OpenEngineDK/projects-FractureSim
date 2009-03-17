#ifndef _SOLID_
#define _SOLID_

#include "VertexPool.h"
#include "Body.h"
#include "Surface.h"

struct TetrahedralTLEDState {
    // global state
	float timeStep;
	float mu, lambda; // should be per vertex or tetra
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

#endif // _SOLID_
