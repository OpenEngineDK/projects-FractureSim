#ifndef _MATERIAL_PROPERTIES_
#define _MATERIAL_PROPERTIES_

class MaterialProperties {
 public:
    float E;
    float nu;
    float density;
    float max_streach;
    float max_compression;

    MaterialProperties(float E, float nu, float density,
                       float max_streach, float max_compression) {
        this->E = E;
        this->nu = nu;
        this->density = density;
        this->max_streach = max_streach;
        this->max_compression = max_compression;
    }
};

#endif // _MATERIAL_PROPERTIES_

