cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0054 NEW)

# --------------- Project configuration ---------------

project (3dns_app LANGUAGES C CXX CUDA)

#--- STATIC or SHARED library ---
option(3DNS_BUILD_SHARED_LIBS "Compile SHARED libraries" OFF)
include(../GcCore/platform.cmake)

#--- Local dependencies ---
set(PROJECT_DEPENDENCIES    "${CMAKE_CURRENT_SOURCE_DIR}/../GcCore/dependencies")
set(PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR ${PROJECT_DEPENDENCIES}/cmake)

# set the output path
set(LIBRARY_OUTPUT_PATH    ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

#--- Add all subdirectories ---
include_directories(../GcCore/api/glm-0.9.9.0/glm)
add_subdirectory(../GcCore/src GcCore)
add_subdirectory(DVR-ray-caster)
