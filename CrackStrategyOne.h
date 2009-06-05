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
    void RenderDebugInfo();

private:
    std::list<int> crackedTetrahedrons;
    std::list<int> crackFront;

    Solid* solid;
    bool crackInitialized;
    int initTetraIdx;
    float4 initPlaneNorm;
    
    


    std::list<float4> cpTri;

    // Crack first tetrahedron 
    void InitializeCrack(Solid* solid, int tetraIndex);
    bool CrackTetrahedron(int tetraIdx, float4 planeNorm, float4 pointOnPlane);

};

#endif _CRACK_STRATEGY_ONE_H_
