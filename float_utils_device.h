#pragma once

#include <vector_functions.h>
#include <device_functions.h>

inline __device__ float2 uintd_to_floatd( uint2 a )
{
	return make_float2( uint2float(a.x), uint2float(a.y) );
}

inline __device__ float3 uint3_to_float3( uint3 a )
{
	return make_float3( uint2float(a.x), uint2float(a.y), uint2float(a.z) );
}

inline __device__ float4 uint4_to_float4( uint4 a )
{
	return make_float4( uint2float(a.x), uint2float(a.y), uint2float(a.z), uint2float(a.w) );
}
