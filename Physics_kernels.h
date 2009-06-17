#include "Solid.h"

void calculateGravityForces(Solid* solid);
void solidCollisionConstraint(Solid* solid, PolyShape* collidableObject);
void applyFloorConstraint(Solid* solid, float floorYPosition);
void calculateInternalForces(Solid* solid, VboManager* vbom);
void updateDisplacement(Solid* solid);
