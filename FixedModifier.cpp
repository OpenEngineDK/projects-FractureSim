
#include "FixedModifier.h"
#include "Physics_kernels.h"


FixedModifier::FixedModifier(PolyShape* bVolume) : Modifier(bVolume) {
    color = make_float4(0.04, 0.6, 0.05, 0.7);
}

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
