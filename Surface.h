#ifndef _SURFACE_
#define _SURFACE_

#include <Meta/CUDA.h>

typedef uint3 Triangle;

class Surface {
 public:
	Triangle* faces;
	unsigned int numFaces;

    Surface();
    Surface(unsigned int numTriangles);
    void Print();
    void ConvertToCuda();
    void DeAlloc();
};

#endif // _SURFACE_
