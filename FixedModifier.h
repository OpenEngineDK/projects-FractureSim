
#ifndef _FIXED_MODIFIER_H_
#define _FIXED_MODIFIER_H_

#include "Modifier.h"

class Solid;
class PolyShape;

class FixedModifier : public Modifier {
public:
    FixedModifier(PolyShape* bVolume);
    ~FixedModifier();

private:
    void ApplyModifierStrategy(Solid* solid);

};

#endif // _FIXED_MODIFIER_H_
