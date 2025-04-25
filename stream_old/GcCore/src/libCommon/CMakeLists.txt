cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (libCommon)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libCommon/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (3DNS_BUILD_SHARED_LIBS)
    # Create a shared library named libCommon
    add_library(Common SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libCommon
    add_library(Common STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Common target
target_link_libraries(Common)

# define an include directory to the Common target
target_include_directories(Common PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Common PUBLIC ${INCLUDE_PATH}/)

# add common compile definitions
target_3dns_definitions(Common)
