
#ifndef _SIMPLE_COLLISION_MODIFIER_H_
#define _SIMPLE_COLLISION_MODIFIER_H_

#include "Modifier.h"

class SimpleCollisionModifier : public Modifier {
public:
    SimpleCollisionModifier(PolyShape* bVolume);
    ~SimpleCollisionModifier();

private:

    void ApplyModifierStrategy(Solid* solid);
};


#endif // _SIMPLE_COLLISION_MODIFIER_H_
