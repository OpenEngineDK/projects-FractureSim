#ifndef _SOLID_FACTORY_
#define _SOLID_FACTORY_

#include "Solid.h"
#include <string>

class SolidFactory {
 public:
    static Solid* Create(std::string name);
};

#endif // _SOLID_FACTORY_
