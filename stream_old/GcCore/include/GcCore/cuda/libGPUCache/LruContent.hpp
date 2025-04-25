#pragma once

#include <cuda.h>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace gpucache
{
    struct TDNS_API LruContent
    {
        uint3 cachePosition;        ///< 3D Position of the corresponding entry in the cache
        uint32_t level;             ///< Resolution level of the block
        float3 volumePosition;      ///< Normalize position of the begining of the block in the volume
    };
} // namespace gpucache
} // namespace tdns