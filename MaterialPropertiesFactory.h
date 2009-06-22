#ifndef _MATERIAL_PROPERTIES_FACTORY_
#define _MATERIAL_PROPERTIES_FACTORY_

#include "MaterialProperties.h"
#include <string>

class MaterialPropertiesFactory {
 public:
    static MaterialProperties* Create(std::string);
};

#endif // _MATERIAL_PROPERTIES_FACTORY_
