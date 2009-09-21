
class Solid;
class VboManager;
class VisualBuffer;

void calculateGravityForces(Solid* solid);
void testCollision(Solid* solid, PolyShape* bVolume, bool* intersect);
void constrainIntersectingPoints(Solid* solid, bool* intersect);
void fixIntersectingPoints(Solid* solid, bool* intersect);
void applyForceToIntersectingNodes(Solid* solid, float3 force, bool* intersect);
void applyDisplacementToIntersectingNodes(Solid* solid, float3 disp, bool* intersect);
void moveIntersectingNodeToSurface(Solid* solid, PolyShape* bVolume, bool* intersect);

void colorSelection(Solid* solid, VisualBuffer* colorBuffer, bool* intersect);

void applyFloorConstraint(Solid* solid, float floorYPosition);
void calculateInternalForces(Solid* solid, VboManager* vbom);
void updateDisplacement(Solid* solid);

void loadArrayIntoVBO(float4* array, unsigned int size, float4* vbo);
