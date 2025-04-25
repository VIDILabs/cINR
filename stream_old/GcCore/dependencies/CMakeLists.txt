cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

include(ExternalProject)

# Target directory for the config-file packages exported by the dependencies
set(DEPENDENCIES_CONFIGFILEPACKAGES_DIR    ${CMAKE_SOURCE_DIR}/cmake)

file(MAKE_DIRECTORY ${DEPENDENCIES_CONFIGFILEPACKAGES_DIR})

if (${CMAKE_VERSION} VERSION_GREATER 3.11.4)
    set(PARALLEL_JOBS_OPTS -j ${BUILD_JOBS})
endif()
set(DEPENDENCIES_BUILD_TYPE Release)
set(DEFAULT_BUILD_COMMAND   ${CMAKE_COMMAND} --build . --config ${DEPENDENCIES_BUILD_TYPE} ${PARALLEL_JOBS_OPTS} -- all)
set(DEFAULT_INSTALL_COMMAND ${CMAKE_COMMAND} --build . --config ${DEPENDENCIES_BUILD_TYPE} ${PARALLEL_JOBS_OPTS} -- install)

###############################################################################
# Dependency build toggle options
###############################################################################
option(BUILD_LZ4        "Build LZ4"         ON)
option(BUILD_GLM        "Build GLM"        OFF)

if (BUILD_LZ4)
    include(lz4.cmake)
endif()

if (BUILD_GLM)
    include(glm.cmake)
endif()
