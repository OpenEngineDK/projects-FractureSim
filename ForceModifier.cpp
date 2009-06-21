
#include "ForceModifier.h"
#include "Physics_kernels.h"
#include "Solid.h"

ForceModifier::ForceModifier(Solid* solid, PolyShape* bVolume, float3 addForce) : 
    Modifier(bVolume), selectionMade(false), addForce(addForce) {
}

ForceModifier::~ForceModifier() {
}

void ForceModifier::SelectNodes(Solid* solid) {
    // Initialize points to be in front of all planes  
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

void ForceModifier::ApplyModifierStrategy(Solid* solid) {
    // Call constraint function
    applyForceToIntersectingNodes(solid, addForce, pIntersect);
   
    if( colorBuffer != NULL && selectionMade ) {
        colorSelection(solid, colorBuffer, pIntersect);
    }
}

