#pragma once

#include "math_def.h"
#include "volume.h"
#include <cuda/cuda_buffer.h>

#ifdef __CUDACC__
#include <random/random.h>
#endif

// #include <GcCore/cuda/libGPUCache/CacheManager.hpp>

namespace ovr {
namespace nncache {

struct PhongMaterial {
  float ambient;
  float diffuse;
  float specular;
  float shininess;
};
using SciVisMaterial = PhongMaterial;

struct LaunchParams { // shared global data
  struct DeviceFrameBuffer {
    vec4f* __restrict__ rgba;
    vec2i size{ 0 };
  } frame;
  vec4f* accumulation;
  int32_t frame_index{ 0 };

  struct DeviceCamera {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera;

  float raymarching_shadow_sampling_scale = 2.f;

  bool enable_sparse_sampling{ false };
  bool enable_path_tracing{ false };
  bool enable_frame_accumulation{ false };

  /* mtls */
  // TODO merge material definitions here
  float scivis_shading_scale = 0.95f;
  PhongMaterial mat_scivis{ .6f, .9f, .4f, 40.f };
  // SciVisMaterial mat_gradient_shading{ .6f, .9f, .4f, 40.f };
  // SciVisMaterial mat_full_shadow{ 1.f, .5f, .4f, 40.f };
  // SciVisMaterial mat_single_shade_heuristic{ 0.8f, .2f, .4f, 40.f };

  /* lights */
  int num_lights = { 1 }; // NOTE: let's only support one directional light here
  struct {
    // vec3f lights_directional_pos[1] = { vec3f(-907.108f, 2205.875f, -400.0267f) };
    // vec3f lights_directional_intensity[1] = { vec3f(1.f) };
    // vec3f light_directional_rgb{ 1.0f };
    // vec3f light_directional_dir{ 0.7f, 0.9f, 0.4f };  
    vec3f direction = vec3f(-907.108f, 2205.875f, -400.0267f);
    vec3f color = vec3f(1.5f);
  } l_distant;
  struct {
    vec3f color = vec3f(1.f);
  } l_ambient;
};

struct OpaqueCacheManager;
std::shared_ptr<OpaqueCacheManager> create_cache_manager(const std::string& config, ValueType type);

#ifdef __CUDACC__
__forceinline__ __device__ bool block_any(bool v) { return __syncthreads_or(v); }
#endif

}
} // namespace ovr
