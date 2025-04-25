#pragma once

#include "array.h"

#include <GcCore/cuda/libGPUCache/CacheManager.hpp>

namespace ovr {
namespace nncache {

struct OpaqueCacheManager {
  friend std::shared_ptr<OpaqueCacheManager>
    create_cache_manager(const std::string& config, const vec3i& capacity, int nbLevels, ValueType type);

public: 
  ValueType type;
  void* cache{nullptr};

  /* CacheManager Metadata */
  uint32_t max_lod;

private:
  using CM_uchar  = tdns::gpucache::CacheManager<uchar1>;
  using CM_ushort = tdns::gpucache::CacheManager<ushort1>;
  using CM_uint   = tdns::gpucache::CacheManager<uint1>;
  using CM_char   = tdns::gpucache::CacheManager<char1>;
  using CM_short  = tdns::gpucache::CacheManager<short1>;
  using CM_int    = tdns::gpucache::CacheManager<int1>;
  using CM_float  = tdns::gpucache::CacheManager<float1>;
  std::unique_ptr<CM_uchar>  m_uchar;
  std::unique_ptr<CM_ushort> m_ushort;
  std::unique_ptr<CM_uint>   m_uint;
  std::unique_ptr<CM_char>   m_char;
  std::unique_ptr<CM_short>  m_short;
  std::unique_ptr<CM_int>    m_int;
  std::unique_ptr<CM_float>  m_float;
};

std::shared_ptr<OpaqueCacheManager> create_cache_manager(
  const std::string& config, 
  const vec3i& capacity, 
  int num_levels, 
  ValueType type
);

std::string get_cache_datapath(const std::string& configFilename);

}
} // namespace ovr
