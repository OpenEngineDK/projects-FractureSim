# Project name. Should be the same as the project directory name and
# will become the name of the binary target.
SET( PROJECT_NAME "TLED")

# cmake_policy(SET CMP0011 OLD) #inlcuded to supress warning

#SET(CMAKE_BUILD_TYPE debug)
#SET(CMAKE_BUILD_TYPE "release")

#SET(CUDA_BUILD_TYPE "Emulation") #can be either: Emulation, Device
SET(CUDA_BUILD_TYPE "Device") #can be either: Emulation, Device

INCLUDE(${CMAKE_SOURCE_DIR}/projects/TLED/conf/cmake/cuda/FindCuda.cmake)

#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}--host-compilation;c")

IF (CMAKE_BUILD_TYPE MATCHES debug)
SET(CUDA_NVCC_FLAGS "-D_DEBUG ${CUDA_NVCC_FLAGS}")
ENDIF (CMAKE_BUILD_TYPE MATCHES debug)

# Add current directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}  
)


MESSAGE("cuda include dir: ${CUDA_INCLUDE}")
MESSAGE("cuda lib dir (cuda and cudart): ${CUDA_TARGET_LINK}")

MESSAGE("cuda sdk include dir: ${CUDA_CUT_INCLUDE}")
MESSAGE("cuda sdk lib dir: ${CUDA_CUT_TARGET_LINK}")

# Use the CUDA_COMPILE macro.
CUDA_INCLUDE_DIRECTORIES(
  "${OE_SOURCE_DIR}"
  ${CUDA_INCLUDE}
  ${CUDA_CUT_INCLUDE} #to include cutil.h
)

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(cuda_kernels
  Physics_kernels.cu
  Visualization_kernels.cu
  Precompute_kernels.cu
  Surface.cu
  Body.cu
  VertexPool.cu
  ${CUDA_TARGET_LINK}
  ${CUDA_CUT_TARGET_LINK}
)
  
INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE} 
  ${CUDA_CUT_INCLUDE} #to include cutil.h
)

# Project source code list
SET( PROJECT_SOURCES
  # Add all the cpp source files here
  main.cpp
  TLEDNode.cpp
  GridNode.cpp
  CoordSystemNode.cpp
  MshObjLoader.cpp
  TetGenLoader.cpp
  VboManager.cpp
  Shapes.cpp
  Precompute.cpp
)

# Include needed to use SDL under Mac OS X
IF(APPLE)
  SET(PROJECT_SOURCES ${PROJECT_SOURCES}  ${SDL_MAIN_FOR_MAC})
ENDIF(APPLE)

# Project executable
ADD_EXECUTABLE(${PROJECT_NAME}
  ${PROJECT_SOURCES}
)

# Project dependencies
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  # Core library dependencies
  OpenEngine_Core
  OpenEngine_Logging
  OpenEngine_Display
  OpenEngine_Devices
  OpenEngine_Scene
  # Extension dependencies
  Extensions_SDL
  Extensions_SetupHelpers
  Extensions_GenericHandlers
  Extensions_OpenGLRenderer
  # Project libs
  ${CUDA_TARGET_LINK}
  ${CUDA_CUT_TARGET_LINK}
  cuda_kernels
)
