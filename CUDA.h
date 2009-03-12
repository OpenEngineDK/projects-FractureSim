#ifndef _CUDA_HEADERS_
#define _CUDA_HEADERS_

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include <Core/Exceptions.h>
#include <Utils/Convert.h>

using namespace OpenEngine;

inline void INITIALIZE_CUDA() {
   cuInit(0);
   //@todo: test that cuda is supported on the platform.
}

inline std::string PRINT_CUDA_DEVICE_INFO() {
    std::string str = "\n";
    int numDevices;
    cuDeviceGetCount(&numDevices);
    str += "number of devices: " + Utils::Convert::ToString(numDevices) + "\n";
    for (int i=0; i<numDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        str += "------------------------\n";
        str += "device: " + Utils::Convert::ToString(i) + "\n";
        str += "name: " + Utils::Convert::ToString(prop.name) + "\n";
        str += "compute capability: " + Utils::Convert::ToString(prop.major) +
            "." + Utils::Convert::ToString(prop.minor) + "\n";
        str += "number of multi processors: " + 
            Utils::Convert::ToString(prop.multiProcessorCount) + "\n";
        str += "supports async memory copy: " + 
            Utils::Convert::ToString(prop.deviceOverlap) + "\n";
        str += "clock rate: " + 
            Utils::Convert::ToString(prop.clockRate) + " kHz\n";
        str += "total memory: " + 
            Utils::Convert::ToString(prop.totalGlobalMem) + " bytes\n";
        str += "global memory: " + 
            Utils::Convert::ToString(prop.totalConstMem) + " bytes\n";
        str += "max shared memory per block: " + 
            Utils::Convert::ToString(prop.sharedMemPerBlock) + " bytes\n";
        str += "max number of registers per block: " + 
            Utils::Convert::ToString(prop.regsPerBlock) + "\n";
        str += "warp size: " + 
            Utils::Convert::ToString(prop.warpSize) + 
            " (cuda executes in half-warps)\n";
        str += "max threads per block: " + 
            Utils::Convert::ToString(prop.maxThreadsPerBlock) + "\n";
        str += "max block size: (" + 
            Utils::Convert::ToString(prop.maxThreadsDim[0]) + "," +
            Utils::Convert::ToString(prop.maxThreadsDim[1]) + "," +
            Utils::Convert::ToString(prop.maxThreadsDim[2]) + ")\n";
        str += "max grid size: (" + 
            Utils::Convert::ToString(prop.maxGridSize[0]) + "," +
            Utils::Convert::ToString(prop.maxGridSize[1]) + "," +
            Utils::Convert::ToString(prop.maxGridSize[2]) + ")\n";
        
        CUdevice dev;
        cuDeviceGet(&dev,i);
        CUcontext ctx;
        cuCtxCreate(&ctx, 0, dev);
        unsigned int free, total;
        cuMemGetInfo(&free, &total);
        str += "total memory: " + 
            Utils::Convert::ToString(total) + " bytes / " +
            Utils::Convert::ToString(free) + " free\n" +
            "total memory: " + 
            Utils::Convert::ToString(((float)total)/1024.0f/1024.0f) +
            " mega bytes / " +
            Utils::Convert::ToString(((float)free)/1024.0f/1024.0f) +
            " free\n";
        cuCtxDetach(ctx);
    }
    str += "------------------------\n";
    return str;
}

/**
 *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
 */
inline void CHECK_FOR_CUDA_ERROR(const std::string file, const int line) {
	cudaError_t errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
        const char* errorString = cudaGetErrorString(errorCode);
        throw Core::Exception("[file:" + file +
                              " line:" + Utils::Convert::ToString(line) +
                              "] CUDA Error: " +
                              std::string(errorString));
	}
}

/**
 *  Checks for CUDA errors and throws an exception if
 *  an error was detected, is only available in debug mode.
 */
//#if OE_DEBUG_GL
#define CHECK_FOR_CUDA_ERROR(); CHECK_FOR_CUDA_ERROR(__FILE__,__LINE__);
//#else
//#define CHECK_FOR_CUDA_ERROR();
//#endif

#endif