
#ifndef _DISPLACEMENT_MODIFIER_H_
#define _DISPLACEMENT_MODIFIER_H_

#include "Modifier.h"

class Solid;
class PolyShape;

class DisplacementModifier : public Modifier {
public:
    DisplacementModifier(Solid* solid, PolyShape* bVolume, float3 addDisplacement);
    ~DisplacementModifier();

    void SelectNodes(Solid* solid);

    bool selectionMade;
    float3 addDisplacement;
       
private:
    void ApplyModifierStrategy(Solid* solid);

};

#endif // _DISPLACEMENT_MODIFIER_H_
