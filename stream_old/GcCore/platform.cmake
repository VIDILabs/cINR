set(TDNS_DEFINITIONS )

if (3DNS_BUILD_SHARED_LIBS)
    message("-- SHARED COMPILATION")
else()
    message("-- STATIC COMPILATION")
    add_definitions(-DTDNS_STATIC)
    list(APPEND TDNS_DEFINITIONS TDNS_STATIC)
endif()

#--- Check des versions des compilateurs ---
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # require at least gcc 5.1
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "GCC version must be at least 5.1!")
    #else() 
    #   set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}"}
    endif()
else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with GCC.")
endif()

#--- Support of C++17 (or C++14 or C++11, check compiler version) ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_CUDA_FLAGS AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # adapted from https://stackoverflow.com/a/69353718
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    if (NOT PROJECT_IS_TOP_LEVEL)
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} PARENT_SCOPE)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Automatically detected GPU architectures: ${CUDA_ARCH_LIST}")
else()
    message("-- CUDA COMPUTE CAPABILITY : ${CMAKE_CUDA_FLAGS}")
endif()

#--- Configure NVCC and CXX flags ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Pass options to NVCC
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --device-debug -g -gencode arch=compute_30,code=sm_30)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g ${CMAKE_CUDA_FLAGS})
    # GCC flags
    set(CMAKE_CXX_FLAGS "-Wall -g -pg")
    # pre-processor define for debug mode
    add_definitions(-DTDNS_DEBUG)
    list(APPEND TDNS_DEFINITIONS TDNS_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Pass options to NVCC // for profiling, add -lineinfo
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 ${CMAKE_CUDA_FLAGS})
    # GCC flags
    set(CMAKE_CXX_FLAGS "-Wall -O3")
    # pre-processor define for release mode
    add_definitions(-DTDNS_RELEASE)
    list(APPEND TDNS_DEFINITIONS TDNS_RELEASE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    # Pass options to NVCC
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --device-debug -g ${CMAKE_CUDA_FLAGS})
    # GCC flags
    set(CMAKE_CXX_FLAGS "-Wall -g -pg -ftest-coverage -fprofile-arcs")
    # pre-processor define for debug mode
    add_definitions(-DTDNS_DEBUG)
    list(APPEND TDNS_DEFINITIONS TDNS_DEBUG)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --extended-lambda ${CMAKE_CUDA_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS})

# pre-processor define for linux
add_definitions(-DTDNS_LINUX)
# add_definitions(-DTDNS_BENCHMARK)
list(APPEND TDNS_DEFINITIONS TDNS_LINUX)

# define functions to add all the definitions
function(target_3dns_definitions target)
    target_compile_definitions(${target} PUBLIC ${TDNS_DEFINITIONS})
endfunction()
