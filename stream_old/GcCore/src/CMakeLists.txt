cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0077 NEW)

# --------------- Project configuration ---------------
project (3dns_core LANGUAGES C CXX CUDA)

#--- STATIC or SHARED library ---
option(3DNS_BUILD_SHARED_LIBS "Compile SHARED libraries" OFF)
include(../platform.cmake)

#--- Local dependencies ---
set(PROJECT_DEPENDENCIES    "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies")
set(PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR ${PROJECT_DEPENDENCIES}/cmake)
find_package(lz4 REQUIRED PATHS ${PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR} NO_DEFAULT_PATH)

# pre-processor define for export libraries compilation
add_definitions(-DTDNS_EXPORT)
list(APPEND TDNS_DEFINITIONS TDNS_EXPORT)

set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../include")

#--- Add all subdirectories ---
add_subdirectory(libCommon)
add_subdirectory(libData)
add_subdirectory(libMath)
add_subdirectory(libNetwork)
add_subdirectory(libPreprocessor)
add_subdirectory(libTinyXml)
add_subdirectory(libPython)

add_subdirectory(cuda/libCommon)
add_subdirectory(cuda/libGPUCache)
add_subdirectory(cuda/libPreprocessor)
