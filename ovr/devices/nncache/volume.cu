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

#include "math_def.h"
#include "volume.h"

#include <fstream>

namespace ovr::nncache {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// void
// StructuredRegularVolume::transform(float transform[12]) const
// {
//   transform[0]  = matrix.l.row0().x;
//   transform[1]  = matrix.l.row0().y;
//   transform[2]  = matrix.l.row0().z;
//   transform[3]  = matrix.p.x;
//   transform[4]  = matrix.l.row1().x;
//   transform[5]  = matrix.l.row1().y;
//   transform[6]  = matrix.l.row1().z;
//   transform[7]  = matrix.p.y;
//   transform[8]  = matrix.l.row2().x;
//   transform[9]  = matrix.l.row2().y;
//   transform[10] = matrix.l.row2().z;
//   transform[11] = matrix.p.z;
// }

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void 
StructuredRegularVolume::set_transform(const affine3f& m)
{
  matrix = m;
}

void
StructuredRegularVolume::set_transfer_function(Array1DFloat4CUDA c, Array1DScalarCUDA a, vec2f r)
{
  device.tfn.colors = c;
  device.tfn.alphas = a;
  set_value_range(r.x, r.y);
  TRACE_CUDA;
}

void
StructuredRegularVolume::set_transfer_function(array_1d_float4_t c, array_1d_scalar_t a, vec2f r)
{
  set_transfer_function(CreateArray1DFloat4CUDA(c), CreateArray1DScalarCUDA(a), r);
}

void
StructuredRegularVolume::set_transfer_function(const std::vector<float>& c, const std::vector<float>& o, const vec2f& r)
{
  tfn_colors_data.resize(c.size() / 3);
  for (int i = 0; i < tfn_colors_data.size(); ++i) {
    tfn_colors_data[i].x = c[3 * i + 0];
    tfn_colors_data[i].y = c[3 * i + 1];
    tfn_colors_data[i].z = c[3 * i + 2];
    tfn_colors_data[i].w = 1.f;
  }
  tfn_alphas_data.resize(o.size() / 2);
  for (int i = 0; i < tfn_alphas_data.size(); ++i) {
    tfn_alphas_data[i] = o[2 * i + 1];
  }

  if (!tfn_colors_data.empty() && !tfn_alphas_data.empty())
  {
    set_transfer_function(CreateArray1DFloat4CUDA(tfn_colors_data), CreateArray1DScalarCUDA(tfn_alphas_data), r);
  }

  TRACE_CUDA;
}

void
StructuredRegularVolume::set_value_range(float data_value_min, float data_value_max)
{
  Array3DScalarCUDA& volume = device.volume;

  // normalize input transfer function value range to the currect floating point 
  // value range, following CUDA's rule:
  //   - if the data type is 8-bit, the value range is [0, 255]
  //   - if the data type is 16-bit, the value range is [0, 65535]
  //   - otherwise, preserve the data value range

  // if (data_value_max >= data_value_min) {
  //   float normalized_max = integer_normalize(data_value_max, volume.type);
  //   float normalized_min = integer_normalize(data_value_min, volume.type);
  //   volume.upper.v = normalized_max; // should use the transfer function value range here
  //   volume.lower.v = normalized_min;    
  // }
  // volume.scale.v = 1.f / (volume.upper.v - volume.lower.v);

  // if (!original_value_range.is_empty()) {
  //   device.tfn.value_range = gdt::intersect(original_value_range, device.tfn.value_range);
  // }

  device.tfn.value_range.lo = 0.f; // = volume.lower.v;
  device.tfn.value_range.hi = 1.f; // = volume.upper.v;
  if (data_value_max > data_value_min) {
    float normalized_min = integer_normalize(data_value_min, volume.type);
    float normalized_max = integer_normalize(data_value_max, volume.type);
    device.tfn.value_range.lo = normalized_min; // should use the transfer function value range here
    device.tfn.value_range.hi = normalized_max;    
  }
  device.tfn.range_rcp_norm = 1.f / device.tfn.value_range.span();

  if (space_partition.allocated()) {
    space_partition.compute_majorant(device.tfn);
  }
}

void
StructuredRegularVolume::set_sampling_rate(float r)
{
  sampling_rate = r;
}

void
StructuredRegularVolume::set_density_scale(float r)
{
  density_scale = r;
}

void
StructuredRegularVolume::commit()
{
  if (!space_partition.allocated()) {
    space_partition.allocate(device.volume.dims);
  }
  if (device.volume.data) {
    space_partition.compute_value_range(device.volume.dims, device.volume.data);
  }

  device.sp = space_partition.device;
  device.transform = matrix;
  device.step = 1.f / sampling_rate;
  device.step_rcp = sampling_rate;
  device.grad_step = vec3f(1.f / vec3f(device.volume.dims));
  device.density_scale = density_scale;

  // check suspecious value range
  if (device.tfn.value_range.empty()) {
    std::cerr << "[warning] invalid value range = " << device.tfn.value_range << std::endl;
  }
}

void 
StructuredRegularVolume::set_space_partition_size(vec3i dims, vec3f spacings)
{
  constexpr int MACROCELL_SIZE = 1 << MACROCELL_SIZE_MIP;
  vec3i mc_dims = dims.long_product() > 0 ? dims : vec3i(MACROCELL_SIZE);
  // vec3i mc_dims = util::div_round_up(device.volume.dims, mc_size);
  // vec3f mc_spacings = vec3f(mc_size) / vec3f(device.volume.dims);
  // std::cout << mc_spacings << std::endl;
  space_partition.allocate(mc_dims, spacings);
}

void
StructuredRegularVolume::load_from_array3d_scalar(array_3d_scalar_t array, float data_value_min, float data_value_max)
{
  Array3DScalarCUDA& output = device.volume;

  if (array->data()) {
    output = CreateArray3DScalarCUDA(array);
    // original_value_range.lo = output.lower.v;
    // original_value_range.hi = output.upper.v;
    // std::cout << "[nncache] volume range = " << original_value_range.lo << " " << original_value_range.hi << std::endl;
  }
  else {
    output.type = array->type;
    output.dims = array->dims;
    // output.lower.v = 0.f;
    // output.upper.v = 1.f;
    // output.scale.v = 1.f / (output.upper.v - output.lower.v);
  }

  set_value_range(data_value_min, data_value_max);
}

}
