#pragma once

#include "array.h"

#ifdef __CUDACC__
#include <random/random.h>
#endif

#include <array>
#include <vector>

#include "core/volumes/macrocell.h"

#define MAX_BYTES_SEND_TO_GPU (10ULL << 30ULL)

namespace ovr {
namespace nncache {

struct SingleSpacePartiton {
public:
  struct Device {
    enum { SIZE_MIP = MACROCELL_SIZE_MIP, MACROCELL_SIZE = 1 << SIZE_MIP };
    range1f* __restrict__ value_ranges{ nullptr };
    float*   __restrict__ majorants{ nullptr };
    vec3i dims; // macrocell dimensions
    vec3f spac;
    vec3f spac_rcp;
    __device__ float access_majorant(const vec3i& cell) const;
  } device;

public: // Changed for loading from params.json
  vnr::MacroCell impl;

  void commit() {
    device.dims = impl.dims();
    device.spac = impl.spacings();
    device.spac_rcp = 1.f / device.spac;
    device.value_ranges = (range1f*)impl.d_value_range();
    device.majorants = (float*)impl.d_max_opacity();
    TRACE_CUDA;
  }

public:
  bool allocated() const { return impl.allocated(); }

  void allocate(vec3i dims, vec3f spac) {
    impl.set_dims(dims);
    impl.set_spacings(spac);
    impl.allocate();
    commit();
  }

  void allocate(vec3i vdims) {
    impl.set_shape(vdims);
    impl.allocate();
    commit();
  }

  void compute_value_range(vec3i dims, cudaTextureObject_t data) { 
    impl.compute_everything(data);
  }

  void compute_majorant(const DeviceTransferFunction& tfn, cudaStream_t stream = 0) {
    impl.update_max_opacity(to_vnr(tfn), stream);
  }
};

inline __device__ float 
SingleSpacePartiton::Device::access_majorant(const vec3i& cell) const
{
  const uint32_t idx = cell.x + cell.y * uint32_t(dims.x) + cell.z * uint32_t(dims.x) * uint32_t(dims.y);
  assert(cell.x < dims.x);
  assert(cell.y < dims.y);
  assert(cell.z < dims.z);
  assert(cell.x >= 0);
  assert(cell.y >= 0);
  assert(cell.z >= 0);
  return majorants[idx];
}

void set_space_partition(SingleSpacePartiton& sp, std::string fname, range1f range, cudaStream_t stream);

} // namespace nncache
} // namespace ovr

namespace tdns {

class Macrocell {
public:
  Macrocell(vnr::vec3i mc_size, vnr::vec3i vol_dims, vnr::ValueType mc_type) : mc_size(mc_size), vol_dims(vol_dims), 
                                                                               mc_type(mc_type) { mc_dims = divRoundUp(vol_dims, mc_size);}
  void process(std::string volume_path, std::string& out_path);

private:
  vnr::vec3i vol_dims;
  vnr::vec3i mc_size;
  vnr::vec3i mc_dims;

  vnr::ValueType mc_type;
}; 

} // namespace tdns
