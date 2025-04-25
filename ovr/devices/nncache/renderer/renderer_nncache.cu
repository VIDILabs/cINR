
#include "renderer.h"
#include "../cachemanager.h"

#include <cuda/cuda_buffer.h>

#include <tiny-cuda-nn/random.h>
using TCNN_NAMESPACE :: generate_random_uniform;
using default_rng_t = TCNN_NAMESPACE :: default_rng_t;

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>

#include <nvtx3/nvtx3.hpp>

using tdns::gpucache::CacheManager;
using tdns::gpucache::VoxelStatus;
using tdns::gpucache::K_CacheManager;
using tdns::gpucache::LruContent;

namespace ovr::nncache {

// --------------------------------------------------------------------------------------------------------
//
// --------------------------------------------------------------------------------------------------------
__device__ float xorshift32f(uint32_t seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;

  return seed / float(UINT32_MAX);
}

template<typename T>
__global__ void sampler_cache_kernel(uint32_t count, K_CacheManager<T> cache, SamplerParams params, vec3f* __restrict__ d_coords, float* __restrict__ d_values, uint32_t seed) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= count) return;

  const auto p = d_coords[i]; // object space position
  float value = d_values[i]; // reads whatever is in the buffer

  // Properly calculating LOD here.
  const float tt = gdt::length(p - params.camera.position) * params.lod_scale;
  // const float tt = params.lod_scale * (std::exp(0.5 * gdt::length(p - params.camera.position)) - std::exp(0.5));
  // const float tt = params.lod_scale * std::pow(max(0.f, gdt::length(p - params.camera.position) - 0.5f), 2);

  const uint32_t base = static_cast<uint32_t>(tt);
  const float prob = tt - base; 

  const uint32_t lod = min(prob > xorshift32f(seed+i) ? base + 1 : base, params.max_lod-1);

  // const uint32_t lod = min(uint32_t(tt * params.lod_scale), params.max_lod);

  // Sample with interpolation
  const auto status = cache.template get_normalized<float>(lod, p, value);
  
  if (status == VoxelStatus::Mapped || status == VoxelStatus::Empty) {
    d_values[i] = value;
  }
  else {
    d_values[i] = 0;
    const uint32_t miss_idx = atomicAdd(params.d_miss_count, 1);
    params.d_missed_coords[miss_idx] = p;
    params.d_missed_indices[miss_idx] = i;
  }
}

// --------------------------------------------------------------------------------------------------------
//
// --------------------------------------------------------------------------------------------------------

template<typename T>
IterativeSampler create_sampler(vnrNetwork net, SamplerParams* params, CacheManager<T>* cache, uint32_t seed=0) {
  return [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
    if (count == 0) return;

    nvtx3::scoped_range nvtx_main{"iterative_lambda_sampler"};

    // stream = nullptr; // WHY: our network implementation uses cuda graph capturing, but causing conflict with our cache implementation.
    // Setting stream to nullptr will force the cuda graph to be disabled.

    const bool enable_network_fallback = params->cachemode == 0;
    const bool enable_network_only     = params->cachemode == 2;

    TRACE_CUDA;

    if (enable_network_only) {
      vnrInfer(net, (vnrDevicePtr)(d_coords), (vnrDevicePtr)(d_values), util::next_multiple(count, 256U), stream);
    }

    else { // mode != 2
      nvtx3::scoped_range nvtx_sampler_cache_kernel{"sampler_cache_kernel"};

      cudaMemsetAsync(params->d_miss_count, 0, sizeof(uint32_t), stream);

      util::linear_kernel(sampler_cache_kernel<T>, 0, stream, count, cache->to_kernel_object(), *params, d_coords, d_values, seed);

      TRACE_CUDA;

      if (enable_network_fallback) { // mode == 0

        nvtx3::scoped_range nvtx_sampler_cache_kernel{"network_fallback"};

        uint32_t h_miss_count;
        cudaMemcpyAsync(&h_miss_count, params->d_miss_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

        const uint32_t n_max = std::min(params->n_train, count);
        const uint32_t n = std::min<uint32_t>(h_miss_count, n_max);

        if (h_miss_count > 0) {
          auto* out_coords = (vec3f*)params->d_missed_coords;
          auto* out_values = (float*)params->d_values_train;
          auto* missed_indices = (uint32_t*)params->d_missed_indices;
          vnrInfer(net, (vnrDevicePtr)(out_coords), (vnrDevicePtr)(out_values), util::next_multiple(n, 256U), stream);
          util::parallel_for_gpu(nullptr, n, [=] __device__ (uint32_t i) {
            const uint32_t idx = missed_indices[i];
            d_values[idx] = out_values[i];
          }); 
        }

      }

      TRACE_CUDA;

    }

    TRACE_CUDA;
  };
}

static default_rng_t rng{ 1337 };

template<typename T>
void render_with_cache(vnrNetwork net, SamplerParams& sparams, CacheManager<T>* cache, std::function<void(IterativeSampler)> callback) 
{
  NVTX3_FUNC_RANGE();

  // Then, rendering
  TRACE_CUDA;
  {
    net = cache->_requestHandler->_bricksManager->_net;
    uint32_t seed = static_cast<uint32_t>(std::rand());
    IterativeSampler sampler = create_sampler(net, &sparams, cache, seed);
    callback(sampler);
  }

  // Next, update cache
  TRACE_CUDA;
  cache->update();

  // Report status
  if (sparams.cachemode == 1) {
    TRACE_CUDA;
    std::vector<float> completude;
    cache->completude(completude); 
    std::string p = std::to_string(completude[0] * 100.f);
    std::cout << "                 - [Cache used " << p.substr(0, p.find('.')+3) << "%] --- \r";
  }

  // Done
  TRACE_CUDA;
}

template void render_with_cache<uchar1> (vnrNetwork, SamplerParams&, CacheManager<uchar1> *, std::function<void(IterativeSampler)>); 
template void render_with_cache<char1>  (vnrNetwork, SamplerParams&, CacheManager<char1>  *, std::function<void(IterativeSampler)>); 
template void render_with_cache<ushort1>(vnrNetwork, SamplerParams&, CacheManager<ushort1>*, std::function<void(IterativeSampler)>); 
template void render_with_cache<short1> (vnrNetwork, SamplerParams&, CacheManager<short1> *, std::function<void(IterativeSampler)>); 
template void render_with_cache<uint1>  (vnrNetwork, SamplerParams&, CacheManager<uint1>  *, std::function<void(IterativeSampler)>); 
template void render_with_cache<int1>   (vnrNetwork, SamplerParams&, CacheManager<int1>   *, std::function<void(IterativeSampler)>); 
template void render_with_cache<float1> (vnrNetwork, SamplerParams&, CacheManager<float1> *, std::function<void(IterativeSampler)>); 

}
