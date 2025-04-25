#pragma once

#include <cstdint>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace graphics
{
    class TDNS_API SDLContext
    {
    public:
        SDLContext(uint32_t flags);

        ~SDLContext();
    };
} // namespace graphics
} // namespace tdns