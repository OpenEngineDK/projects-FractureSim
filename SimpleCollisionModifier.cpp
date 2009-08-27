
#include "SimpleCollisionModifier.h"
#include "Physics_kernels.h"
#include "Solid.h"

SimpleCollisionModifier::SimpleCollisionModifier(PolyShape* bVolume) : Modifier(bVolume) {}

SimpleCollisionModifier::~SimpleCollisionModifier() {}

void SimpleCollisionModifier::ApplyModifierStrategy(Solid* solid) {
    // Initialize points to be in front of all planes  
    CudaMemset(pIntersect, true, sizeof(bool)*solid->vertexpool->size);

    // Test intersection with bounding volume
    testCollision(solid, bVolume, pIntersect);

    // Call constraint function
    constrainIntersectingPoints(solid, pIntersect);
}
