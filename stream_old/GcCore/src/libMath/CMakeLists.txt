cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (libMath)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libMath/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (3DNS_BUILD_SHARED_LIBS)
    # Create a shared library named libMath
    add_library(Math SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libMath
    add_library(Math STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# define an include directory to the Math target
target_include_directories(Math PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Math PUBLIC ${INCLUDE_PATH}/)

# add common compile definitions
target_3dns_definitions(Math)
