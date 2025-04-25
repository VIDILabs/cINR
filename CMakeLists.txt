# ======================================================================== #
# Copyright 2019-2022 Qi Wu                                                #
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

project(cINR LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.18)

# ------------------------------------------------------------------------------

include(configure.cmake)
add_subdirectory(base)

# ------------------------------------------------------------------------------

include(ExternalProject)

option(OVR_BUILD_MODULE_NNCACHE  "Build NN Cache Device" ON)

if(OVR_BUILD_MODULE_NNCACHE)

  if(DEFINED GDT_CUDA_ARCHITECTURES) 
    message(STATUS "Obtained target architecture from environment variable GDT_CUDA_ARCHITECTURES=${GDT_CUDA_ARCHITECTURES}")
    set(ENV{TCNN_CUDA_ARCHITECTURES} ${GDT_CUDA_ARCHITECTURES})
  endif()

  ################
  # tiny-cuda-nn #
  ################
  if(NOT TARGET tiny-cuda-nn)

    # Directly download tcnn from github
    FetchContent_Declare(tcnn_content
      GIT_REPOSITORY "https://github.com/wilsonCernWq/tiny-cuda-nn.git"
      GIT_TAG "5bd6b38dead69e74f7233ec69a8f3bc043ae2a1d"
    )
    FetchContent_GetProperties(tcnn_content)
    if (NOT tcnn_content_POPULATED)
        FetchContent_Populate(tcnn_content)
    endif()
    set(TCNN_SOURCE_DIR ${tcnn_content_SOURCE_DIR})

    # Build it as a submodule
    set(TCNN_BUILD_BENCHMARK OFF)
    set(TCNN_BUILD_EXAMPLES  OFF)
    add_subdirectory(${TCNN_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/tcnn_build EXCLUDE_FROM_ALL)
    target_compile_definitions(tiny-cuda-nn INTERFACE TCNN_NAMESPACE=tcnn)
    if(UNIX)
      set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
      set_target_properties(tiny-cuda-nn PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

  endif()
  add_library(tcnn ALIAS tiny-cuda-nn)

  # Simulate include paths
  include_directories(ovr/${CMAKE_CURRENT_LIST_DIR})

  ################
  #
  ################
  set(ENABLE_OUT_OF_CORE ON CACHE BOOL "Enable Out-of-Core" FORCE)
  set(ENABLE_OPENGL ${OVR_BUILD_OPENGL} CACHE BOOL "Enable OpenGL apps" FORCE)
  add_subdirectory(ovr/core)

  ################
  #
  ################
  set(3DNS_BUILD_SHARED_LIBS ON CACHE BOOL "GcCore Build shared libraries" FORCE)
  include_directories(stream/GcCore/api/glm-0.9.9.0/glm)
  include_directories(stream/GcCore/include)
  add_subdirectory(stream/GcCore/src gccore)

  ################
  #
  ################
  add_subdirectory(ovr/devices/nncache)
  add_subdirectory(ovr/apps)

  # Profiling
  add_subdirectory(NVTX/c)

endif()
