#pragma once

#include <exception>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    /**
    * @def CUDA_SAFE_CALL(fct)
    * @brief Check if an error occurs in the CUDA called function.
    *
    * @param fct    CUDA function to call.
    *
    * @throw std::runtime_error if an error occurs.
    */
#if TDNS_MODE == TDNS_MODE_DEBUG || TDNS_MODE == TDNS_MODE_RELEASE //Both modes for the moment
#define CUDA_SAFE_CALL(status) do                                       \
    {                                                                   \
        if (status != cudaSuccess)                                      \
        {                                                               \
            std::stringstream error;                                    \
            error << "CUDA error (file \"" << __FILE__                  \
            << "\": line " << std::to_string(__LINE__)                  \
            << " - error = [" << cudaGetErrorString(status) << "].";    \
            throw std::runtime_error(error.str().c_str());              \
        }                                                               \
    } while(false)
#define CUDA_SAFE_CALL_NOEXCEPT(status) do                              \
    {                                                                   \
        if (status != cudaSuccess)                                      \
        {                                                               \
            std::cerr << "CUDA error (file \"" << __FILE__              \
            << "\": line " << std::to_string(__LINE__)                  \
            << " - error = [" << cudaGetErrorString(status) << "].";    \
        }                                                               \
    } while(false)
#else
#define CUDA_SAFE_CALL(status) do { } while(false)
#define CUDA_SAFE_CALL_NOEXCEPT(status) do { } while(false)
#endif

    /**
    * @def CUDA_CHECK_KERNEL_ERROR()
    * @brief Check if the last kernel call leads to an error.
    *
    * @throw std::runtime_error if an error occurs.
    */
#if TDNS_MODE == TDNS_MODE_DEBUG || TDNS_MODE == TDNS_MODE_RELEASE //Both modes for the moment
#define CUDA_CHECK_KERNEL_ERROR() do                                    \
    {                                                                   \
        cudaError_t cuError = cudaGetLastError();                       \
        if (cuError != cudaSuccess)                                     \
        {                                                               \
            std::stringstream error;                                    \
            error << "CUDA error (file \"" << __FILE__                  \
            << "\": line " << std::to_string(__LINE__)                  \
            << " - error = [" << cudaGetErrorString(cuError) << "].";   \
            throw std::runtime_error(error.str().c_str());              \
        }                                                               \
    } while(false)
#else
#define CUDA_CHECK_KERNEL_ERROR(status) do { } while(false)
#endif
} // namespace common
} // namespace tdns