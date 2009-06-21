
#include "SimpleCollisionModifier.h"
#include "Physics_kernels.h"


SimpleCollisionModifier::SimpleCollisionModifier(PolyShape* bVolume) : Modifier(bVolume) {}

SimpleCollisionModifier::~SimpleCollisionModifier() {}

void SimpleCollisionModifier::ApplyModifierStrategy(Solid* solid) {
   // Test intersection with bounding volume
    testCollision(solid, bVolume, pIntersect);

    // Call constraint function
    constrainIntersectingPoints(solid, pIntersect);
}
