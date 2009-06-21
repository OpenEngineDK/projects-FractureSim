
#include "FixedModifier.h"
#include "Physics_kernels.h"


FixedModifier::FixedModifier(PolyShape* bVolume) : Modifier(bVolume) {}

FixedModifier::~FixedModifier() {}

void FixedModifier::ApplyModifierStrategy(Solid* solid) {
   // Test intersection with bounding volume
    testCollision(solid, bVolume, pIntersect);

    // Call constraint function
    fixIntersectingPoints(solid, pIntersect);

    if( colorBuffer != NULL ) {
        colorSelection(solid, colorBuffer, pIntersect);
    }
}
