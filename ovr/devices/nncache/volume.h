//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#pragma once

#include "array.h"
#include "space_partition.h"

#include <array>
#include <vector>

namespace ovr {
namespace nncache {

// ------------------------------------------------------------------
// Volume Definition
// ------------------------------------------------------------------

// ------------------------------------------------------------------
//
// Host Functions
//
// ------------------------------------------------------------------

struct StructuredRegularVolume {
public:
  using DeviceData = Array3DScalarCUDA;
  using DeviceSingleSpacePartiton = SingleSpacePartiton::Device;

  struct Device {
    DeviceData volume;
    DeviceSingleSpacePartiton sp;
    DeviceTransferFunction tfn;

    float step = 1.f;
    float density_scale = 1.f;
    box3f bbox = box3f(vec3f(0), vec3f(1)); // with respect to [0-1]^3
    affine3f transform;

    // GPU cacne to avoid recomputation
    float step_rcp = 1.f; 
    vec3f grad_step;
  };

  Device device;

  float sampling_rate = 1.f;
  float density_scale = 1.f;

public: // Public for params.json macroCell loading
  affine3f matrix;
  std::vector<vec4f> tfn_colors_data;
  std::vector<float> tfn_alphas_data;
  // range1f original_value_range;
  SingleSpacePartiton space_partition;

#if defined(__cplusplus)

public:
  void commit();
  void load_from_array3d_scalar(array_3d_scalar_t array, float data_value_min = 1, float data_value_max = -1);
  void set_space_partition_size(vec3i dims, vec3f spacings);
  void set_sampling_rate(float r);
  void set_density_scale(float r);
  void set_transform(const affine3f& m);
  void set_transfer_function(Array1DFloat4CUDA c, Array1DScalarCUDA a, vec2f r);
  void set_transfer_function(array_1d_float4_t c, array_1d_scalar_t a, vec2f r);
  void set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const vec2f& r);
  void set_value_range(float data_value_min, float data_value_max);

#endif // #if defined(__cplusplus)
};

using DeviceGrid = StructuredRegularVolume::Device;


struct VNRDeviceVolume {
  Array3DScalarCUDA volume;
  DeviceTransferFunction tfn;

  float step_rcp = 1.f;
  float step = 1.f;
  float density_scale = 1.f;

  vec3f grad_step;

  float* __restrict__ macrocell_max_opacity{ nullptr };
  vec2f* __restrict__ macrocell_value_range{ nullptr };
  vec3i macrocell_dims;
  vec3f macrocell_spacings;
  vec3f macrocell_spacings_rcp;

  box3f bbox = box3f(vec3f(0), vec3f(1)); // object space box

  affine3f transform;
};

} // namespace nncache
} // namespace ovr