#ifndef _PRECALCULATE_
#define _PRECALCULATE_

#include "Solid.h"

void moveAccordingToBoundingBox(Solid* solid);

float precompute(Solid* solid, 
                float smallestAllowedVolume, 
                float smallestAllowedLength,
                float timeStepFactor, float damping);

void createNeighbourList(Solid* solid);

#endif // _PRECLCULATE_
