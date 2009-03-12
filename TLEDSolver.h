#include "TetrahedralMesh.h"

// pre
void precompute(Solid* solid, 
                float density, float smallestAllowedVolume, 
                float smallestAllowedLength,
                float mu, float lambda, float timeStepFactor, float damping); 
//Body* loadMesh(const char* filename);
//Surface* loadSurfaceOBJ(const char* filename);


// physics
void calculateGravityForces(Solid* solid); 
void doTimeStep(Solid* solid);
void applyFloorConstraint(Solid* solid, float floorYPosition); 


// rendering
void display(unsigned int object_number, Solid* solid, float4* buf);

// helper functions
void cleanupDisplay(void);
float CPUPrecalculation(Solid* solid, int blockSize, int& return_maxNumForces,
                        float density, float smallestAllowedVolume,
                        float smallestAllowedLength);
void calculateInternalForces(Solid* solid);
