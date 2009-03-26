#ifndef _CUDA_HEADERS_
#define _CUDA_HEADERS_

#include "../extensions/OpenGLRenderer/Meta/OpenGL.h"

#include <cuda.h>
#include <cutil.h>
#include <cufft.h>

#include <vector_types.h>
#include <cutil_math.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h> // includes cudaMalloc and cudaMemset

#include <driver_types.h> // includes cudaError_t

#include <cuda_gl_interop.h>

//#include <cutil_gl_error.h>
//@todo: generetes linker errors, but includes CUT_CHECK_ERROR_GL()

using namespace OpenEngine;

inline void INITIALIZE_CUDA() {
    cuInit(0);
    //@todo: test that cuda is supported on the platform.

    //@todo: print installed cuda version, and info like opengl.

    #ifdef _DEBUG
    printf("CUDA_SAFE_CALL: enabled\n");
    #else
    printf("CUDA_SAFE_CALL: disabled\n");
    #endif
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
 
inline void THROW_ERROR(const char* file, const int line,
                        const char* errorString) {
    const int bLength = 256;
    char buffer[bLength];
    int n = sprintf (buffer,"[file: %s line: %i] CUDA Error: %s\n",
                     file, line,
                     errorString);
    printf( buffer );
    if (n < 0)
        printf("error when writing error CUDA message\n");
    if (n >= bLength)
        printf("error message buffer was to small\n");
    exit(-1);
}

/**
 *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
 *  inspired by cutil.h: CUT_CHECK_ERROR
 */
inline void CHECK_FOR_CUDA_ERROR(const char* file, const int line) {
    cudaError_t errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess) {
        const char* errorString = cudaGetErrorString(errorCode);
        THROW_ERROR(file, line, errorString);
    }
    errorCode = cudaThreadSynchronize();
    if (errorCode != cudaSuccess) { 
        const char* errorString = cudaGetErrorString(errorCode);
        THROW_ERROR(file, line, errorString);
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
