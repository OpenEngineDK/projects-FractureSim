
#ifndef _MOVABLE_COLLISION_MODIFIER_H_
#define _MOVABLE_COLLISION_MODIFIER_H_

#include "Modifier.h"

class MovableCollisionModifier : public Modifier {
public:
    MovableCollisionModifier(PolyShape* bVolume);
    ~MovableCollisionModifier();

private:

    void ApplyModifierStrategy(Solid* solid);
};


#endif // _MOVABLE_COLLISION_MODIFIER_H_
