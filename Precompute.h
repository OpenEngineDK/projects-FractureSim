#ifndef _PRECALCULATE_
#define _PRECALCULATE_

#include "Solid.h"

void moveAccordingToBoundingBox(Solid* solid);

void precompute(Solid* solid, 
                float density, float smallestAllowedVolume, 
                float smallestAllowedLength, float mu,
                float lambda, float timeStepFactor, float damping);

void createNeighbourList(Solid* solid);

#endif // _PRECLCULATE_
