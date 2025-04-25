#pragma once

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/cuda/libGPUCache/CacheManager.hpp>
#include <GcCore/libData/MetaData.hpp>

namespace tdns
{
namespace gpucache
{
    template<typename T>
    class CacheManager;

} // namespace gpucache
} // namespace tdns

namespace tdns
{
namespace graphics
{
    /**
    * @brief
    */
    template<typename T>
    void TDNS_API display_volume_raycaster(tdns::gpucache::CacheManager<T>  *manager, tdns::data::MetaData &volumeData);

} // namespace graphics
} // namespace tdns