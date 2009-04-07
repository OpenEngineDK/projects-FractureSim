#ifndef _COLOR_RAMP_
#define _COLOR_RAMP_

#include <Meta/CUDA.h>

//from: http://local.wasp.uwa.edu.au/~pbourke/texture_colour/colourramp/

/*
   Return a RGB colour value given a scalar v in the range [vmin,vmax]
   In this case each colour component ranges from 0 (no contribution) to
   1 (fully saturated), modifications for other ranges is trivial.
   The colour is clipped at the end of the scales if v is outside
   the range [vmin,vmax]
*/

__device__
inline float4 GetColor(double v, double vmin, double vmax) {
    float4 c = make_float4(1.0f, 1.0f, 1.0f, 1.0f); // white, full alpha
    double dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c.x = 0;
        c.y = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c.x = 0;
        c.z = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c.x = 4 * (v - vmin - 0.5 * dv) / dv;
        c.z = 0;
    } else {
        c.y = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c.z = 0;
    }
    return c;
}

#endif // _COLOR_RAMP_
