
#ifndef _FORCE_MODIFIER_H_
#define _FORCE_MODIFIER_H_

#include "Modifier.h"

class Solid;
class PolyShape;

class ForceModifier : public Modifier {
public:
    ForceModifier(Solid* solid, PolyShape* bVolume, float3 addForce);
    ~ForceModifier();

    void SelectNodes(Solid* solid);
    void SetForce(float3 force);

    float3 addForce;       
private:
    bool selectionMade;


    void ApplyModifierStrategy(Solid* solid);

};

#endif // _FORCE_MODIFIER_H_
