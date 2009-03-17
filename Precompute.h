#ifndef _PRECALCULATE_
#define _PRECALCULATE_

#include "Solid.h"

void precompute(Solid* solid, 
                float density, float smallestAllowedVolume, 
                float smallestAllowedLength, float mu,
                float lambda, float timeStepFactor, float damping);

#endif // _PRECLCULATE_
