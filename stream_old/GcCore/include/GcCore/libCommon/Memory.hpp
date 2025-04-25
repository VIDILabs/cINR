#pragma once

#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    /*
     * make_unique() c++14 features does not work for a library build with NVCC on LINUX
     */
    template<typename T, typename... Args>
    std::unique_ptr<T> create_unique_ptr(Args &&... args)
    {
        #if TDNS_OS == TDNS_OS_WINDOWS
            return std::make_unique<T>(std::forward<Args>(args)...);
        #elif TDNS_OS == TDNS_OS_LINUX
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        #else
            static_assert(false, "OS not supported !");
        #endif
    }
} // namespace app
} // namespace tdns