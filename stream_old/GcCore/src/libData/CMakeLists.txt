cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (libData)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libData/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (3DNS_BUILD_SHARED_LIBS)
    # Create a shared library named libData
    add_library(Data SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libData
    add_library(Data STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Data target
target_link_libraries(Data
    Common
    Math
    TinyXml
    lz4
)

# define an include directory to the Data target
target_include_directories(Data PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Data PUBLIC ${INCLUDE_PATH}/)

# add common compile definitions
target_3dns_definitions(Data)
