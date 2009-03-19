#include "Solid.h"

void calculateGravityForces(Solid* solid);
void applyFloorConstraint(Solid* solid, float floorYPosition);
void calculateInternalForces(Solid* solid, VboManager* vbom);
void updateDisplacement(Solid* solid);
