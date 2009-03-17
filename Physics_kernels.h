#include "Solid.h"

void calculateGravityForces(Solid* solid);
void applyFloorConstraint(Solid* solid, float floorYPosition);
void calculateInternalForces(Solid* solid);
void updateDisplacement(Solid* solid);
