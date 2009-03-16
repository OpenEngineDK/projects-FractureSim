# Project name. Should be the same as the project directory name and
# will become the name of the binary target.
SET( PROJECT_NAME "TLED")

SET(CMAKE_BUILD_TYPE debug)
#SET(CMAKE_BUILD_TYPE "release")

#SET(CUDA_BUILD_TYPE "Emulation") #can be either: Emulation, Device
SET(CUDA_BUILD_TYPE "Device") #can be either: Emulation, Device

INCLUDE(${CMAKE_SOURCE_DIR}/projects/TLED/conf/cmake/cuda/FindCuda.cmake)

#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}--host-compilation;c")


# Add current directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  
)

# Use the CUDA_COMPILE macro.
CUDA_INCLUDE_DIRECTORIES(
  "${OE_SOURCE_DIR}"
  ${FOUND_CUDA_NVCC_INCLUDE}
  "/Developer/CUDA/common/inc/" #to include cutil.h
  ${GLUT_INCLUDE_DIR}
)

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(cuda_lib
  FEM_kernels.cu
  ImageD.cu
  TLEDSolver.cu
  Visualization_kernels.cu
  TetrahedralMesh.cu
  ${CUDA_CUT_TARGET_LINK}
)
  
INCLUDE_DIRECTORIES(
  ${FOUND_CUDA_NVCC_INCLUDE}
  "/Developer/CUDA/common/inc/" #to include cutil.h
)

# Project source code list
SET( PROJECT_SOURCES
  # Add all the cpp source files here
  ${CUDA_CUT_TARGET_LINK}
  ${CUDA_TARGET_LINK}
  main.cpp
  TLEDNode.cpp
  GridNode.cpp
  CoordSystemNode.cpp
  MshObjLoader.cpp
  TetGenLoader.cpp
  VboManager.cpp
  Shapes.cpp
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
  ${CUDA_TARGET_LINK}
  cuda_lib
)
