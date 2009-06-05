#include "Shapes.h"

#ifndef _VISUALIZATION_KERNELS_H_
#define _VISUALIZATION_KERNELS_H_

class Solid;
class VboManager;

void applyTransformation(VisualBuffer& vert, VisualBuffer& norm);

// rendering
void updateSurface(Solid* solid, VboManager* vbom);
void updateCenterOfMass(Solid* solid, VboManager* vbom);
void updateBodyMesh(Solid* solid, VboManager* vbom, float minX);
void updateStressTensors(Solid* solid, VboManager* vbom);
void planeClipping(Solid* solid, VboManager* vbom, float minX);


#endif // _VISUALIZATION_KERNELS_H_
