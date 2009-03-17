#ifndef _PRECOMPUTE_KERNELS_
#define _PRECOMPUTE_KERNELS_

#include "Solid.h"

void precalculateABC(float timeStep, float damping, VertexPool* vertexpool);
void precalculateShapeFunctionDerivatives(Solid* solid);

#endif // _PRECOMPUTE_KERNELS_
