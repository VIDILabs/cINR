#pragma once

#include <string>
#include <vector>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace data
{
    struct TDNS_API CacheConfiguration
    {
        std::vector<tdns::math::Vector3ui> CacheSize;           ///< Sizes of the caches. 0 is the data caches.
        std::vector<tdns::math::Vector3ui> BlockSize;           ///< Size of the blocks in the data caches. 0 is the brick size.

        uint32_t DataCacheFlags;                                ///< Flags for the data cache (e.g. Normalize access or not).
    };
} // namespace data
} // namespace tdns