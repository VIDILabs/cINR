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

#include "renderer.h"
#include "../cachemanager.h"
#include <core/volumes/volumes.h>
#include <cuda/cuda_buffer.h>

#include <iostream>

#ifdef ENABLE_LOGGING
#define log() std::cout
#else
static std::ostream null_output_stream(0);
#define log() null_output_stream
#endif

using tdns::gpucache::CacheManager;
using tdns::gpucache::VoxelStatus;
using tdns::gpucache::K_CacheManager;

namespace ovr::nncache {

// --------------------------------------------------------------------------------------------------------
//
// --------------------------------------------------------------------------------------------------------

const uint32_t MAX_INFERENCE_SIZE = 1 << 24;

void network_inference_kernel(vnr::NeuralVolume* network, vec3f* __restrict__ d_coords, float* __restrict__ d_values, uint32_t count, cudaStream_t stream) {
  for (uint32_t i = 0; i < count; i += MAX_INFERENCE_SIZE) {
    // 'sample_coord' and 'sample_value' are allocated with padding to the next multiple of 256
    const uint32_t batch = util::next_multiple(std::min(count - i, MAX_INFERENCE_SIZE), 256U);
    network->inference(batch, (float*)(d_coords+i), d_values+i, stream);
  }
}

void groundtruth_inference_kernel(const VNRDeviceVolume& volume, vec3f* __restrict__ d_coords, float* __restrict__ d_values, uint32_t count, cudaStream_t stream) {
  util::parallel_for_gpu(stream, count, [=] __device__ (uint32_t i) {
    const auto p = d_coords[i];
    d_values[i] = tex3D<float>(volume.volume.data, p.x, p.y, p.z);
  });
}

// NOTE: this function is defined in renderer_nncache.cu
template<typename T>
void render_with_cache(vnrNetwork net, SamplerParams& params, CacheManager<T>* cache, std::function<void(IterativeSampler)> cb);

// --------------------------------------------------------------------------------------------------------
//
// --------------------------------------------------------------------------------------------------------

void RenderObject::init(affine3f transform, 
  ValueType type, vec3i dims, // range1f range, 
  vec3i macrocell_dims, 
  vec3f macrocell_spacings, 
  vec2f* macrocell_d_value_range, 
  float* macrocell_d_max_opacity
  // std::string params_path
) {
  self.transform = transform;
  self.volume.dims = dims;
  self.volume.type = type;
  self.macrocell_value_range = macrocell_d_value_range;
  self.macrocell_max_opacity = macrocell_d_max_opacity;
  self.macrocell_dims = macrocell_dims;
  self.macrocell_spacings = macrocell_spacings;
  self.macrocell_spacings_rcp = 1.f / macrocell_spacings;

  net = nullptr;
}

void RenderObject::update(int rendering_mode, 
  const DeviceTransferFunction& tfn,
  float sampling_rate,
  float density_scale,
  vec3f clip_lower, 
  vec3f clip_upper,
  const vnr::Camera& camera,
  const vec2i& framesize
) {
  if (this->rendering_mode != rendering_mode) {
    this->rendering_mode = rendering_mode;
    rm.clear(stream);
    pt.clear(stream);
  }

  self.step = 1.f / sampling_rate;
  self.step_rcp = sampling_rate;
  self.grad_step = vec3f(1.f / vec3f(self.volume.dims));
  self.density_scale = density_scale;
  self.tfn = tfn;
  self.tfn.range_rcp_norm = 1.f / self.tfn.value_range.span();
  self.bbox.lower = clip_lower;
  self.bbox.upper = clip_upper;

  // resize our cuda frame buffer
  params.frame.size = framesize;
  framebuffer_accumulation.resize(framesize.long_product() * sizeof(vec4f), stream);
  params.accumulation = (vec4f*)framebuffer_accumulation.d_pointer();

  /* camera ... */
  /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
  const float fovy = camera.fovy;
  const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * (float)M_PI / 180.f);
  const float aspect = params.frame.size.x / float(params.frame.size.y);
  params.camera.position = camera.from;
  params.camera.direction = normalize(camera.at - camera.from);
  params.camera.horizontal = t * aspect * normalize(cross(params.camera.direction, camera.up));
  params.camera.vertical = cross(params.camera.horizontal, params.camera.direction) / aspect;

  // flag to reset frame data
  params.frame_index = 0;
}

void RenderObject::render(vec4f* fb, const IterativeSampler& sampler, bool iterative) {
  constexpr auto PT = VNR_PATHTRACING_SAMPLE_STREAMING;
  constexpr auto RM_B = VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING;
  constexpr auto RM_G = VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING;
  constexpr auto RM_S = VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING;
  constexpr auto mRMB = MethodRayMarching::NO_SHADING;
  constexpr auto mRMG = MethodRayMarching::GRADIENT_SHADING;
  constexpr auto mRMS = MethodRayMarching::SINGLE_SHADE_HEURISTIC;
  if (params.frame.size.x == 0 || params.frame.size.y == 0) return;
  params.frame_index++;
  params.frame.rgba = fb;
  switch (rendering_mode) {
  case PT: pt.render(stream, params, self, sampler, iterative); break;
  case RM_B: rm.render(stream, params, self, sampler, mRMB, iterative); break;
  case RM_G: rm.render(stream, params, self, sampler, mRMG, iterative); break;
  case RM_S: rm.render(stream, params, self, sampler, mRMS, iterative); break;
  default: std::cerr << "WARNING: unknown rendering mode " << rendering_mode << std::endl; break;
  }
}

void RenderObject::render(vec4f* fb, vnr::NeuralVolume* network, cudaTextureObject_t grid) {
  self.volume.data = grid; // set volume data
  if (!network && self.volume.data == 0) { std::cerr << "WARNING: no volume data to render" << std::endl; return; }
  if (network) {
    render(fb, [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
      network_inference_kernel(network, d_coords, d_values, count, stream);
    }, /*iterative=*/true);
  }
  else {
    render(fb, [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
      groundtruth_inference_kernel(self, d_coords, d_values, count, stream);
    }, /*iterative=*/false);
  }
}

void RenderObject::render(vec4f* fb, vnr::NeuralVolume* _unused, OpaqueCacheManager& manager) {
  SamplerParams sparams;

  const auto wto = self.transform.inverse();
  sparams.camera.position   = xfmPoint (wto, vec3f(params.camera.position));
  sparams.camera.direction  = xfmVector(wto, vec3f(params.camera.direction));
  sparams.camera.horizontal = xfmVector(wto, vec3f(params.camera.horizontal));
  sparams.camera.vertical   = xfmVector(wto, vec3f(params.camera.vertical));

  // const uint32_t batchsize = 1 << 16;
  const uint32_t n_pixels = util::next_multiple<size_t>(params.frame.size.long_product(), 256U) * 16;
  // const uint32_t n_train  = std::max(batchsize*16, util::next_multiple(n_pixels, batchsize));
  // TODO: using static buffer to hack it for now, should think of a better way
  static CUDABuffer train_coords, train_values, missed_coords, missed_indices, miss_count;
  {
    // status.resize(n_pixels * sizeof(uint8_t), stream);
    miss_count.resize(sizeof(uint32_t), stream);
    missed_indices.resize(n_pixels * sizeof(uint32_t), stream);
  }
  { 
    missed_coords.resize(n_pixels * sizeof(vec3f), stream);
    // train_coords.resize(n_pixels * sizeof(vec3f), stream);
    train_values.resize(n_pixels * sizeof(float), stream);
  }
  // sparams.d_status = (uint8_t*)status.d_pointer();
  sparams.d_miss_count = (uint32_t*)miss_count.d_pointer();
  sparams.d_missed_indices = (uint32_t*)missed_indices.d_pointer();
  sparams.d_missed_coords = (vec3f*)missed_coords.d_pointer();
  sparams.d_coords_train = (vec3f*)train_coords.d_pointer();
  sparams.d_values_train = (float*)train_values.d_pointer();
  sparams.n_train = n_pixels;
  sparams.n_curr = 0;

  sparams.value_range = self.tfn.value_range;
  sparams.cachemode = cachemode;
  sparams.max_lod = manager.max_lod;
  // sparams.lod_scale = lod.start_lod_scale;
  sparams.lod_threshold = lod.threshold;

  lod.start_lod_scale = lod.scale; // Comment this out to enable lod-preloading

  // TODO: can possibly pass the new net_id here then through to the CM to reload the net in the BM (the Macrocell structure won't be reloaded this way however)
  if (lod.start_lod_scale != lod.scale) {
    sparams.lod_scale = lod.start_lod_scale;
    lod.start_lod_scale += (lod.start_lod_scale - lod.scale) > 0 ? -0.005 : 0.005;
  }
  else 
    sparams.lod_scale = lod.start_lod_scale;

  auto cb = [&] (IterativeSampler sampler) { render(fb, sampler); };

  TRACE_CUDA;

  // switch (manager.type) {
  // case VALUE_TYPE_UINT8:  render_with_cache(net, sparams, (CacheManager<uchar1> *)manager.cache, cb); break;
  // case VALUE_TYPE_INT8:   render_with_cache(net, sparams, (CacheManager<char1>  *)manager.cache, cb); break;
  // case VALUE_TYPE_UINT16: render_with_cache(net, sparams, (CacheManager<ushort1>*)manager.cache, cb); break;
  // case VALUE_TYPE_INT16:  render_with_cache(net, sparams, (CacheManager<short1> *)manager.cache, cb); break;
  // case VALUE_TYPE_UINT32: render_with_cache(net, sparams, (CacheManager<uint1>  *)manager.cache, cb); break;
  // case VALUE_TYPE_INT32:  render_with_cache(net, sparams, (CacheManager<int1>   *)manager.cache, cb); break;
  // case VALUE_TYPE_FLOAT:  render_with_cache(net, sparams, (CacheManager<float1> *)manager.cache, cb); break;
  // default: throw std::runtime_error("unsupported type encountered: " + std::to_string(manager.type));
  // }
  render_with_cache(net, sparams, (CacheManager<float1> *)manager.cache, cb);
}

} // namespace vnr
