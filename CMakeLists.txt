# Project name. Should be the same as the project directory name and
# will become the name of the binary target.
SET( PROJECT_NAME "TLED")
SET( CUDA_LIB "CUDA_KERNELS")

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY( ${CUDA_LIB}
  Physics_kernels.cu
  Visualization_kernels.cu
  Precompute_kernels.cu
  CudaMath.cu
)

# Project source code list
SET( PROJECT_SOURCES
  # Add all the cpp source files here
  main.cpp
  TLEDNode.cpp
  VertexPool.cpp
  Surface.cpp
  Solid.cpp
  SolidFactory.cpp
  MaterialPropertiesFactory.cpp
  Body.cpp
  GridNode.cpp
  CoordSystemNode.cpp
  MshObjLoader.cpp
  TetGenLoader.cpp
  VboManager.cpp
  Shapes.cpp
  Precompute.cpp
  CrackStrategyOne.cpp
  CudaMem.cpp
  Modifier.cpp
  FixedModifier.cpp
  ForceModifier.cpp
  SimpleCollisionModifier.cpp
)

# Include needed to use SDL under Mac OS X
IF(APPLE)
  SET(PROJECT_SOURCES ${PROJECT_SOURCES}  ${SDL_MAIN_FOR_MAC})
ENDIF(APPLE)

# Project executable
ADD_EXECUTABLE( ${PROJECT_NAME}
  ${PROJECT_SOURCES}
)

# Project dependencies
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
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
  ${CUDA_TARGET_LINK} #cuda
  ${CUDA_CUT_TARGET_LINK} #cutil
  ${CUDA_LIB}
)
