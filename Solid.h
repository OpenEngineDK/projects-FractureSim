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

    Solid();
    ~Solid();

    void DeAlloc();
    void Print();
    bool IsInitialized();

};

#endif // _SOLID_
