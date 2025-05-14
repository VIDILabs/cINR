# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(APPLE) # MacOS is not supported ...
	set(CMAKE_MACOSX_RPATH ON)
endif()
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
else()
	# if(BUILD_SHARED_LIBS)
	# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  #   set(CMAKE_POSITION_INDEPENDENT_CODE ON)
	# endif()
endif()
# if(NOT WIN32)
#   # visual studio doesn't like these (not need them):
#   set(CMAKE_CXX_FLAGS "--std=c++17")
#   set(CUDA_PROPAGATE_HOST_FLAGS ON)
# endif()
# if(UNIX)
#   set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	# if(BUILD_SHARED_LIBS)
	# 	list(APPEND CUDA_NVCC_FLAGS "-fPIC")
	# endif()
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_CONVERSIONS__")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF2_OPERATORS__")

if(TRUE)

  # adapted from https://stackoverflow.com/a/69353718
  find_package(CUDA REQUIRED)
  # enable_language(CUDA)

  if(DEFINED GDT_CUDA_ARCHITECTURES) 
    message(STATUS "Obtained target architecture from environment variable GDT_CUDA_ARCHITECTURES=${GDT_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_ARCHITECTURES ${GDT_CUDA_ARCHITECTURES})
  endif()

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Automatically detected GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  
  # we can only enable CUDA if CMAKE_CUDA_ARCHITECTURES is set
  if (CMAKE_CUDA_ARCHITECTURES)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "Cannot utomatically detected GPU architecture")
  endif()

endif()

set(EXTERN_DIR ${CMAKE_CURRENT_LIST_DIR}/base/extern)

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
include_directories(${EXTERN_DIR})
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

include(${EXTERN_DIR}/bin2c/target_add_embeded_shaders.cmake)

# ------------------------------------------------------------------
# load external system libraries
# ------------------------------------------------------------------
find_package(Threads REQUIRED)
# find_package(OpenMP  REQUIRED)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${EXTERN_DIR}/gdt/cmake")
include(configure_build_type)
include_directories(${EXTERN_DIR}/gdt)

# ------------------------------------------------------------------
# find OpenGL
# ------------------------------------------------------------------
if(OVR_BUILD_OPENGL)

  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  if(TARGET OpenGL::OpenGL)
    list(APPEND GFX_LIBRARIES OpenGL::OpenGL)
  else()
    list(APPEND GFX_LIBRARIES OpenGL::GL)
  endif()
  if(TARGET OpenGL::GLU)
    list(APPEND GFX_LIBRARIES OpenGL::GLU)
  endif()
  if(TARGET OpenGL::GLX)
    list(APPEND GFX_LIBRARIES OpenGL::GLX)
  endif()

  # list(APPEND GFX_LIBRARIES glfw)
  # list(APPEND GFX_LIBRARIES glad)
  # list(APPEND GFX_LIBRARIES imgui)
  # list(APPEND GFX_LIBRARIES glfwApp)

endif()

# ------------------------------------------------------------------
# import CUDA
# ------------------------------------------------------------------
if(OVR_BUILD_CUDA)  
  include(configure_cuda)
  mark_as_advanced(CUDA_SDK_ROOT_DIR)
endif()

# ------------------------------------------------------------------
# import Optix7
# ------------------------------------------------------------------
if(OVR_BUILD_OPTIX7)  
  include(configure_optix)
endif(OVR_BUILD_OPTIX7)

# ------------------------------------------------------------------
# import OneAPI
# ------------------------------------------------------------------
find_package(TBB REQUIRED)

if(OVR_BUILD_OSPRAY)
  find_package(ospray 2.0 REQUIRED)
endif(OVR_BUILD_OSPRAY)

# if(OVR_BUILD_OPENVKL)
#   find_package(rkcommon REQUIRED)
#   find_package(openvkl REQUIRED)
# endif(OVR_BUILD_OPENVKL)
