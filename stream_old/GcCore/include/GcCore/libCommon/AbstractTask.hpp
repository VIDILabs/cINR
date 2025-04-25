#pragma once

#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    class TDNS_API AbstractTask
    {
    public:

        /**
        * @brief Destructor
        */
        virtual ~AbstractTask() = default;
        
        /**
        * @brief virtual method to do task
        */
        virtual void do_task() = 0;

        /**
        * @brief getter of delta time
        *
        * @return delta time
        */
        uint64_t  deltaTime() { return _deltaTime; }
    protected:
        uint64_t  _deltaTime; ///< periodic time
    };
} // common
} // tdns