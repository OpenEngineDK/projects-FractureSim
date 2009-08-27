
#include "MovableCollisionModifier.h"
#include "Physics_kernels.h"
#include "Solid.h"

MovableCollisionModifier::MovableCollisionModifier(PolyShape* bVolume) : Modifier(bVolume) {}

MovableCollisionModifier::~MovableCollisionModifier() {}

void MovableCollisionModifier::ApplyModifierStrategy(Solid* solid) {
    // Initialize points to be in front of all planes  
    CudaMemset(pIntersect, true, sizeof(bool)*solid->vertexpool->size);

    // Test intersection with bounding volume
    testCollision(solid, bVolume, pIntersect);

    // Call constraint function
    constrainIntersectingPoints(solid, pIntersect);

    // Call constraint function
    moveIntersectingNodeToSurface(solid, bVolume, pIntersect);
}
