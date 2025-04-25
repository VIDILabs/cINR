#pragma once

#include <cstdint>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace graphics
{
    void TDNS_API create_sdl_context(uint32_t flags);

    void TDNS_API quit_sdl();
} // namespace graphics
} // namespace tdns

