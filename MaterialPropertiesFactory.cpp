#include "MaterialPropertiesFactory.h"

#include <Core/Exceptions.h>
#include <Logging/Logger.h>

using namespace OpenEngine;

#define KPa 1000.0f;
#define MPa 1000.0f * KPa;
#define GPa 1000.0f * MPa;

MaterialProperties* MaterialPropertiesFactory::Create(std::string name) {
    logger.info << "Loading material properties: " << name << logger.end;

    float E = 0.0; // Modulus of elasticity (Young's modulus)
    float nu = 0.0; // Poisson's ratio

	float density;
    float mu;
    float lambda;

    // todo
    float max_streach;
    float max_compression;
    
    if (name == "") {
        throw Core::Exception("please enter material properties");
    } else if (name == "soft") {
        density = 0.001f;
        mu = 80007.0f;
        lambda = 49329.0f;
        max_streach = 6000;
        max_compression = max_streach;
    } else if (name == "stiff") {
        density = 2400; // kg / m^3
        E = 20 * GPa;
        nu = 0.21;
        float factor = 1; //50 virker
        max_streach = 5 * factor * MPa; // Tensile strength
        max_compression = 40 * MPa; // Compressive strength
    } else if (name == "steel") {
        density = 7860; // kg / m^3
        E = 200 * GPa;
        nu = 0.21;
        float factor = 1; //50 virker
        max_streach = 0;
        max_compression = max_streach;
    }else if (name == "yelly") {
        density = 240; // kg / m^3
        E = 210 * MPa;
        nu = 0.21;
        float factor = 1; //50 virker
        max_streach = 0;
        max_compression = max_streach;
    } else if (name == "dentine") {
        //from: http://www.engineeringtoolbox.com/concrete-properties-d_1223.html
        density = 2400; // kg / m^3
        E = 15 * GPa;
        nu = 0.21;
        float factor = 35; //50 virker
        max_streach = 5 * factor * MPa; // Tensile strength
        max_compression = 40 * factor * MPa; // Compressive strength
    } else if (name == "concrete") {
        //from: http://www.engineeringtoolbox.com/concrete-properties-d_1223.html
        density = 2400; // kg / m^3
        E = 41 * GPa;
        nu = 0.21;
        float factor = 35; //50 virker
        max_streach = 5 * factor * MPa; // Tensile strength
        max_compression = 40 * factor * MPa; // Compressive strength
    } else if (name == "concrete_moded") {
        density = 2.4f;
        mu = 136.36 * GPa;
        lambda = 8.334 * GPa;
        max_streach = 3 * MPa; // Pa
        max_compression = max_streach * 10;
    } else if (name == "glass") {
        density = 2450.0f;
        E = 71.0f * GPa;
        float G = 30.0f * GPa; // shear modulus
        nu = (E/(2*G)) - 1;
        max_streach = 3500 * MPa; // s
        max_compression = max_streach;
    } else if (name == "rubber") {
        density = 0.001f;
        mu = 207.0f;
        lambda = 2500.0f;
        max_streach = 15 * MPa;
        max_compression = max_streach;
    } else if (name == "iron") {
        density = 7810;
        E = 207 * GPa;
        float G = 140.0f * GPa; // shear modulus
        nu = (E/(2*G)) - 1;
        max_streach = 2310 * MPa;
        max_compression = max_streach;
    } else if (name == "cement") {
        density = 2010;
        E = 11.2 * GPa;
        nu = 0.21;
        max_streach = 0.910 * MPa;
        max_compression = max_streach;
    } else
        throw Core::Exception("unknown material properties");
    

    // mu and lambda is Lame's constants used for calculating
    // Young's modulus and Poisson's ratio.

    // E is Young's modulus, it is a material constant defining axial
    // deformations within the material. 
    // It simply tells how much a material will stretch or compress
    // when forces are applied.
    // [Ref: FEM, Smith, page 661, C.8a]
    // [Ref: Fysik-bog, Erik, page. 168];
    if( E == 0.0)
        E = mu*(3.0f*lambda+2.0f*mu)/(lambda+mu);

    // mu is Poisson's ratio. 
    // (From wiki) When a sample of material is stretched in one direction, it
    // tends to contract (or rarely, expand) in the other two
    // directions. Conversely, when a sample of material is compressed
    // in one direction, it tends to expand in the other two
    // directions. Poisson's ratio (ν) is a measure of this tendency.
    // [Ref: FEM, Smith, page 661, C.8b]
    // [Ref: Fysik-bog, Erik, page. 171]
    if (nu == 0.0)
        nu = lambda/(2.0f*(lambda+mu));


    logger.info << "Material Properties - E: "
                << E << ", G: " << nu << logger.end;
    return new MaterialProperties(E,nu,density,max_streach,max_compression);
}
