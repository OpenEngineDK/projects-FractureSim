// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------


#ifndef _CRACK_STRATEGY_ONE_H_
#define _CRACK_STRATEGY_ONE_H_

#include "CrackStrategy.h"
#include "Solid.h"
#include <list>

class CrackStrategyOne : public CrackStrategy {
public:
    CrackStrategyOne();
    ~CrackStrategyOne();

    bool CrackInitialized(Solid* solid);
    void InitializeCrack(Solid* solid);
    void ApplyCrackTracking(Solid* solid);
    bool FragmentationDone();

    // Debug
    void RenderDebugInfo(Solid* solid);


private:
    std::list<int> crackedTetrahedrons;
    std::list<int> crackFront;
    bool crackInitialized;
    int exceedCount;
    int initTetraIdx;
    float3 initPlaneNorm;

    
    // Crack first tetrahedron 
    void InitializeCrack(Solid* solid, int tetraIndex);
    bool CrackTetrahedron(Solid* solid, int tetraIdx, float3 planeNorm, float3 pointOnPlane);

    int GetCrackedNeighbour(Solid* solid, int tetraIdx);


    // Debug
    std::list<float3> debugPlaneNorm;
    std::list<float3> debugPointOnPlane;    
    std::list<float3> debugVector;
    void AddDebugVector(float3 pos, float3 dir, float3 color);

};

#endif _CRACK_STRATEGY_ONE_H_
