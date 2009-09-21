
#include "DisplacementModifier.h"
#include "Physics_kernels.h"
#include "Solid.h"

DisplacementModifier::DisplacementModifier(Solid* solid, PolyShape* bVolume, float3 addDisplacement) : 
    Modifier(bVolume), selectionMade(false), addDisplacement(addDisplacement) {
    color = make_float4(0.06, 0.62, 0.98, 0.7);
}

DisplacementModifier::~DisplacementModifier() {
}

void DisplacementModifier::SelectNodes(Solid* solid) {
    // Initialize points to be in front of all planes (clear intersection)  
    CudaMemset(pIntersect, true, sizeof(bool)*solid->vertexpool->size);

    // Test intersection with bounding volume
    testCollision(solid, bVolume, pIntersect);

    if( colorBuffer != NULL ) {
        CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&colorBuffer->buf, colorBuffer->vboID));
        colorSelection(solid, colorBuffer, pIntersect);
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject( colorBuffer->vboID ));
        selectionMade = true;
    }
}

void DisplacementModifier::ApplyModifierStrategy(Solid* solid) {
    // Call constraint function
    fixIntersectingPoints(solid, pIntersect);

    // Call constraint function
    if( selectionMade )
        applyDisplacementToIntersectingNodes(solid, addDisplacement, pIntersect);

    if( colorBuffer != NULL && selectionMade ) {
        colorSelection(solid, colorBuffer, pIntersect);
    }
}
