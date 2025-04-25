cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (libPython)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libPython/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (3DNS_BUILD_SHARED_LIBS)
    # Create a shared library named libPython
    add_library(Python SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libPython
    add_library(Python STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Python target
target_link_libraries(Python
    Common
    Data
    Preprocessor
    Math
)

# define an include directory to the Python target
target_include_directories(Python PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Python PUBLIC ${INCLUDE_PATH}/)

# add common compile definitions
target_3dns_definitions(Python)
