cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (libcudaPreprocessor LANGUAGES CXX CUDA)

# find_package(CUDA QUIET REQUIRED)

# --------------- Library configuration ---------------

# CAREFUL : DEPRECATED !! (used for CUDA) (cuda_include_directories() doesn't works)
include_directories(${INCLUDE_PATH}/)

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/cuda/libPreprocessor/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

# if (3DNS_BUILD_SHARED_LIBS)
#     # Create a shared library named libPreprocessor
#     add_library(cudaPreprocessor SHARED 
#         ${HEADER_FILES}
#         ${SOURCE_FILES}
#     )
# else()
#     # Create a static library named libPreprocessor
#     add_library(cudaPreprocessor STATIC 
#         ${HEADER_FILES}
#         ${SOURCE_FILES}
#     )
# endif()

# Create a static library named libPreprocessor
add_library(cudaPreprocessor STATIC 
    ${HEADER_FILES}
    ${SOURCE_FILES}
)
if (3DNS_BUILD_SHARED_LIBS)
    set_property(TARGET cudaPreprocessor PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
set_property(TARGET cudaPreprocessor PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# link libraries to the Preprocessor target
target_link_libraries(cudaPreprocessor PRIVATE
    Common
    Math
    Data
)

# define an include directory to the Preprocessor target
target_include_directories(cudaPreprocessor PRIVATE ${INCLUDE_PATH}/)
target_include_directories(cudaPreprocessor PUBLIC ${INCLUDE_PATH}/)
target_compile_options(cudaPreprocessor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# add common compile definitions
target_3dns_definitions(cudaPreprocessor)
