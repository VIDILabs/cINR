#pragma once

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Utility class that makes any derived class non-copyable.
    */
    class TDNS_API Noncopyable
    {
    protected:

        /**
        * @brief Default constructor. Here to force the compiler to create one.
        */
        Noncopyable() = default;

        /**
        * @brief Default destructor
        *
        * By declaring a protected destructor it's impossible to call delete on
        * a pointer of NonCopyable, thus preventing possible resource leaks.
        */
        ~Noncopyable() = default;

    private:
        /**
        * @brief Disable the copy constructor.
        */
        Noncopyable(const Noncopyable &) = delete;

        /**
        * @brief Disabled assignment operator.
        */
        Noncopyable& operator = (const Noncopyable &) = delete;
    };
} // namespace common
} // namespace tdns