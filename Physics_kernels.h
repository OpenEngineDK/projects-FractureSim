
class Solid;
class VboManager;

void calculateGravityForces(Solid* solid);
void solidCollisionConstraint(Solid* solid, PolyShape* collidableObject, bool* intersect);
void constrainIntersectingPoints(Solid* solid, bool* intersect);
void applyFloorConstraint(Solid* solid, float floorYPosition);
void calculateInternalForces(Solid* solid, VboManager* vbom);
void updateDisplacement(Solid* solid);

void loadPolyShapeIntoVBO(PolyShape* obj, float4* buf);
