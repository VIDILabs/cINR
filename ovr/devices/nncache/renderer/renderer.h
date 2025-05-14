//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
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

#include "api.h"
#include "methods.h"

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <vector>

namespace vnr {
  struct NeuralVolume;
}

namespace ovr::nncache {

struct SamplerParams {
  // We can use this to pass additional parameters to the sampler.
  // uint8_t*  __restrict__ d_status;
  float*    __restrict__ d_values_train;
  vec3f*    __restrict__ d_coords_train;
  vec3f*    __restrict__ d_missed_coords;
  uint32_t* __restrict__ d_missed_indices;
  uint32_t* __restrict__ d_miss_count;
  uint32_t n_train = 0;
  uint32_t n_curr = 0;

  // Other parameters
  LaunchParams::DeviceCamera camera; // in world space
  range1f value_range; // TODO not used for now
  int cachemode = 0;

  // Or other simple values
  uint32_t max_lod;
  float lod_scale;
  float lod_threshold;
};

// ------------------------------------------------------------------
// renderer functions
// ------------------------------------------------------------------

struct RenderObject {
  LaunchParams params;
  VNRDeviceVolume self;

  MethodRayMarching rm;
  MethodPathTracing pt;
  MethodMegakernel megakernel;

  int cachemode = 0;
  struct {
    float scale = 1.f;
    float threshold = 1.f;
    float start_lod_scale = 5.0f;
  } lod;

  // --------------------------------------------------------------- //
  // --------------------------------------------------------------- //
  int rendering_mode{ VNR_INVALID };
  CUDABuffer framebuffer_accumulation;
  cudaStream_t stream{ nullptr };

  vnrNetwork net;

  void init(
    affine3f transform, 
    ValueType type, vec3i dims, // range1f range, 
    vec3i macrocell_dims, 
    vec3f macrocell_spacings, 
    vec2f* macrocell_d_value_range, 
    float* macrocell_d_max_opacity
    // std::string params_path
  );

  void update(int rendering_mode, 
    const DeviceTransferFunction& tfn,
    float sampling_rate,
    float density_scale,
    vec3f clip_lower, 
    vec3f clip_upper,
    const vnr::Camera& camera,
    const vec2i& framesize
  );

  void render(vec4f* fb, const IterativeSampler& sampler, bool iterative = true);
  void render(vec4f* fb, vnr::NeuralVolume* neuralnet, cudaTextureObject_t grid);
  void render(vec4f* fb, vnr::NeuralVolume* neuralnet, OpaqueCacheManager& manager);
};

// void pytorch_set_param(float time);
// void pytorch_set_param(float theta, float phi);
// void pytorch_inference_kernel(cudaStream_t stream, uint32_t count, vnr::vec3f* __restrict__ d_coords, float* __restrict__ d_values);

} // namespace vnr
