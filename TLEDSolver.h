#include "TetrahedralMesh.h"

void precompute(TetrahedralMesh* mesh, TetrahedralTLEDState *state, 
						   float density, float smallestAllowedVolume, float smallestAllowedLength,
   						   float mu, float lambda, float timeStepFactor, float damping); 
void doTimeStep(TetrahedralMesh* mesh, TetrahedralTLEDState *state);
TetrahedralMesh* loadMesh(const char* filename);
void display(unsigned int object_number, TetrahedralMesh* mesh, TetrahedralTLEDState *state, TriangleSurface* surface, float4* bufArray);
void display(unsigned int object_number, TetrahedralMesh* mesh, TetrahedralTLEDState *state, TriangleSurface* surface, float4** bufArray);
TriangleSurface* loadSurfaceOBJ(const char* filename);
void calculateGravityForces(TetrahedralMesh* mesh, TetrahedralTLEDState *state); 
void applyFloorConstraint(TetrahedralMesh* mesh, TetrahedralTLEDState *state, float floorYPosition); 
void cleanupDisplay(void);
float CPUPrecalculation(TetrahedralMesh *mesh, int blockSize, int& return_maxNumForces, float density, float smallestAllowedVolume, float smallestAllowedLength);
void calculateInternalForces(TetrahedralMesh* mesh, TetrahedralTLEDState *state);
