
#include "Shapes.h"

#ifndef _VISUALIZATION_KERNELS_H_
#define _VISUALIZATION_KERNELS_H_

class Solid;
class VboManager;

void applyTransformation(VisualBuffer& vb);

// rendering
void updateSurface(Solid* solid, VboManager* vbom);
void updateCenterOfMass(Solid* solid, VboManager* vbom);
void updateStressTensors(Solid* solid, VboManager* vbom);

#endif // _VISUALIZATION_KERNELS_H_
