cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (libNetwork)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libNetwork/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (3DNS_BUILD_SHARED_LIBS)
    # Create a shared library named libNetwork
    add_library(Network SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libNetwork
    add_library(Network STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Network target
target_link_libraries(Network
    Common
    Math
    Data
)

# define an include directory to the Network target
target_include_directories(Network PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Network PUBLIC ${INCLUDE_PATH}/)

# add common compile definitions
target_3dns_definitions(Network)
