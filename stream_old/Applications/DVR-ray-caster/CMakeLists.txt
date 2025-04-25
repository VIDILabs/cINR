cmake_minimum_required(VERSION 3.8)

# --------------- Project configuration ---------------

project (main LANGUAGES CUDA)

# find_package(CUDA QUIET REQUIRED)
# since cmake 3.11, set the cmake policy for opengl preference to choose LEGACY to avoid Warnings
set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
# pre-processor define for GLEW
add_definitions(-DGLEW_STATIC)
find_package(Threads REQUIRED)

# CAREFUL : DEPRECATED !! (used for CUDA) (cuda_include_directories() doesn't works)
include_directories(
    include/
    ${GLEW_INCLUDE_DIRS}
)

link_libraries(${GLEW_LIBRARIES})

file(GLOB_RECURSE HEADER_FILES "include/*.hpp" "include/*.h")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.cu")

# Create an executable
add_executable(main
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

# link a library
target_link_libraries(main
    Common
    Data
    Math
    TinyXml
    Preprocessor
    cudaCommon
    cudaPreprocessor
    cudaGPUCache
    ${OPENGL_gl_LIBRARY}
    ${GLEW_LIBRARIES}
    SDL2
    Threads::Threads
)

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(main PRIVATE include)
target_include_directories(main PUBLIC  ${PROJECT_GCCORE}/include/)
target_compile_definitions(main PUBLIC  TDNS_DEBUG)

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Copy the "config" directory content in a config directory at the same level than binaries
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
                       ${CMAKE_SOURCE_DIR}/../config $<TARGET_FILE_DIR:main>/config)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/../assets/shaders $<TARGET_FILE_DIR:main>/shaders)
