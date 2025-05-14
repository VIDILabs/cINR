//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "methods.h"
#include "raytracing.h"
#include "dda.h"

#include <cuda/cuda_buffer.h>

#ifndef ADAPTIVE_SAMPLING
#define ADAPTIVE_SAMPLING 1
#endif
// #define ADAPTIVE_SAMPLING 0

namespace ovr::nncache {

using random::RandomTEA;

inline int 
initialize_N_ITERS() 
{
  int n_iters = 16;
  if (const char* env_p = std::getenv("VNR_RM_N_ITERS")) {
    n_iters = std::stoi(env_p);
  }
  return n_iters;
}

const int N_ITERS = initialize_N_ITERS();
// constexpr int N_ITERS = 16;
// constexpr int N_ITERS = 64; // fps = 6.668
// constexpr int N_ITERS = 32; // fps = 7.200
// constexpr int N_ITERS = 16; // fps = 7.249
// constexpr int N_ITERS = 8; // fps = 7.073
// constexpr int N_ITERS = 4; // fps = 6.611
// constexpr int N_ITERS = 2; // fps = 5.936
// constexpr int N_ITERS = 1; // fps = 4.786

using ShadingMode = MethodRayMarching::ShadingMode;
constexpr auto RMB = MethodRayMarching::NO_SHADING;
constexpr auto RMG = MethodRayMarching::GRADIENT_SHADING;
constexpr auto RMS = MethodRayMarching::SINGLE_SHADE_HEURISTIC;
constexpr auto SHADOW = MethodRayMarching::SHADOW;

struct Ray;

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<bool ADAPTIVE>
struct RayMarchingIter;

template<>
struct RayMarchingIter<true> : private dda::DDAIter {
public:
  using DDAIter::cell;
  using DDAIter::t_next;
  using DDAIter::next_cell_begin;
public:
  __device__ RayMarchingIter() {}
  __device__ RayMarchingIter(const VNRDeviceVolume& self, const Ray& ray);
  __device__ bool
  resumable(const VNRDeviceVolume& self, const Ray& ray);

  template<typename F>
  __device__ void 
  exec(const VNRDeviceVolume& self, const Ray& ray, 
       const float step, const uint32_t pidx, const F& body);
};

template<>
struct RayMarchingIter<false>{
public:
  float next_cell_begin{};
private:
  uint32_t __pad[2];
public:
  __device__ RayMarchingIter() {}
  __device__ RayMarchingIter(const VNRDeviceVolume& self, const Ray& ray);
  __device__ bool
  resumable(const VNRDeviceVolume& self, const Ray& ray);

  template<typename F>
  __device__ void 
  exec(const VNRDeviceVolume& self, const Ray& ray, 
       const float step, const uint32_t pidx, const F& body);
};

using Iter = RayMarchingIter<ADAPTIVE_SAMPLING>;



// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<typename F>
__device__ void
raymarching_iterator_0(const VNRDeviceVolume& self, 
                       const vec3f& org, const vec3f& dir,
                       const float tMin, const float tMax, 
                       const float step, 
                       const F& body, 
                       bool debug);

template<typename F>
__device__ void
raymarching_iterator_1(const VNRDeviceVolume& self, 
                       const vec3f& org, const vec3f& dir,
                       const float tMin, const float tMax, 
                       const float step, 
                       const F& body, 
                       bool debug);

template<bool ADAPTIVE, typename F>
__device__ void
raymarching_iterator(const VNRDeviceVolume& self, const Ray& ray,
                     const float step, const F& body, bool debug = false)
{
  if constexpr (ADAPTIVE) {
    raymarching_iterator_0(self, ray.org, ray.dir, ray.tnear, ray.tfar, step, body, debug);
  }
  else {
    raymarching_iterator_1(self, ray.org, ray.dir, ray.tnear, ray.tfar, step, body, debug);
  }
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

struct SingleShadePayload
{
  vec3f highest_org = 0.f;
  vec3f highest_color = 0.f;
  float highest_alpha = 0.f;
};

struct PackedPayload {
  // 0 //
  vec3f color_or_org;
  float alpha;
  // 16 //
  uint32_t pidx = 0;
  float jitter = 0.f;
  Iter iter;
  SingleShadePayload ss;
};

struct Ray {
  vec3f org{};
  vec3f dir{};
  float tnear = 0.f;
  float tfar = float_large;

  uint32_t pidx = 0;
  float jitter = 0.f;
  Iter iter;
};

template<ShadingMode MODE> 
struct RayPayload : Ray {
  float alpha = 0.f;
  vec3f color = 0.f;
};

template<> 
struct RayPayload<RMS> : Ray {
  float alpha = 0.f;
  vec3f color = 0.f;
  SingleShadePayload ss;
};

template<> 
struct RayPayload<SHADOW> : Ray {
  float alpha = 0.f;
};

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

using Packet = PacketTemplate<PackedPayload>;
static_assert((sizeof(PackedPayload)) % sizeof(uint4) == 0, "Incorrect size of PackedRay");

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

#include "raymarching_iter.inc"

struct RayMarchingData : LaunchParams
{
  RayMarchingData(const LaunchParams& p) : LaunchParams(p) {}

  ShadingMode mode;

  VNRDeviceVolume volume;

  vec3f* __restrict__ inference_coords{ nullptr };
  float* __restrict__ inference_values{ nullptr };

  // belows are only useful for sampling streaming
  uint8_t* flags{ nullptr };

  // per ray payload (ordered by ray index) //
  Packet::DataType* packets[Packet::N] = {};

  // ordered by pixel index //
  vec4f* __restrict__ final_org_and_jitter{ nullptr };
  vec4f* __restrict__ final_highest_rgba{ nullptr };
  vec4f* __restrict__ shading_rgba{ nullptr };
};

struct RayMarchingIterData {
  Packet packet;
};

struct RayMarchingIterView {
  typedef RayMarchingIterView IterView;
  typedef RayMarchingIterData IterData;

  Packet::DataType* packets[Packet::N];

  __host__ __device__ RayMarchingIterView() {}

  __host__ RayMarchingIterView(const RayMarchingData& params, uint32_t offset) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      this->packets[i] = params.packets[i] + offset;
    }
  }

  __host__ __device__ void move(ptrdiff_t n) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      this->packets[i] += n;
    }
  }

  __host__ __device__ ptrdiff_t diff(const IterView& other) const {
    return this->packets[0] - other.packets[0];
  }

  /// Conversion operator
  __host__ __device__ operator IterData() const {
    IterData value;
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      value.packet.data[i] = *this->packets[i];
    }
    return value;
  }

  /// Assignment operator
  __host__ __device__ IterView& operator=(const IterData& value) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      *this->packets[i] = value.packet.data[i];
    }
    return *this;
  }
};

using IterView = RayMarchingIterView;
using IterData = RayMarchingIterData;

// NOTE: what is the best SoA layout here?

__forceinline__ __device__ bool intersectVolume(Ray& ray, const VNRDeviceVolume& self) {
  return intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self);
}

template<ShadingMode MODE> 
inline __device__ void 
compute_ray(const RayMarchingData& params, RayPayload<MODE>& ray)
{
  const auto& fbIndex = ray.pidx;

  // compute pixel ID
  const uint32_t ix = fbIndex % params.frame.size.x;
  const uint32_t iy = fbIndex / params.frame.size.x;

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f& otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  // generate ray direction
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
}

template<> 
inline __device__ void 
compute_ray<SHADOW>(const RayMarchingData& params, RayPayload<SHADOW>& ray)
{
  // get the object to world transformation
  const affine3f& otw = params.volume.transform;
  const affine3f wto = otw.inverse();
  // generate ray direction
  ray.dir = xfmVector(wto, normalize(params.l_distant.direction));
}

template<ShadingMode MODE> 
inline __device__ RayPayload<MODE> 
load(const RayMarchingData& params, const uint32_t ridx) 
{
  Packet packet;

  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    packet.data[i] = params.packets[i][ridx];
  }

  RayPayload<MODE> ray;
  ray.pidx = packet.items.pidx;
  ray.jitter = packet.items.jitter;
  ray.alpha = packet.items.alpha;
  if constexpr (MODE == SHADOW) {
    ray.org = packet.items.color_or_org;
  } else {
    ray.color = packet.items.color_or_org;
  }
  if constexpr (MODE == RMS) {
    ray.ss = packet.items.ss;
  }
  ray.iter = packet.items.iter;

  compute_ray(params, ray);

  const bool hashit = intersectVolume(ray, params.volume);
  assert(hashit);

  return ray;
}

template<ShadingMode MODE> 
inline __device__ void
save(const RayMarchingData& params, const RayPayload<MODE>& ray, const uint32_t ridx)
{
  Packet packet;
  packet.items.pidx = ray.pidx;
  packet.items.jitter = ray.jitter;
  packet.items.alpha = ray.alpha;
  if constexpr (MODE == SHADOW) {
    packet.items.color_or_org = ray.org;
  } else {
    packet.items.color_or_org = ray.color;
  }
  if constexpr (MODE == RMS) {
    packet.items.ss = ray.ss;
  }
  packet.items.iter = ray.iter;

  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    params.packets[i][ridx] = packet.data[i];
  }

  params.flags[ridx] = 0xFF;
}

/* standard version */ void
do_raymarching_trivial(cudaStream_t stream, const RayMarchingData& params);

/* iterative version */ void
do_raymarching_iterative(cudaStream_t stream, const RayMarchingData& params, const IterativeSampler& sampler, uint32_t numPixels);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T>
inline T* 
define_buffer(char* begin, size_t& offset, size_t buffer_size)
{
  auto* ret = (T*)(begin + offset); 
  offset += buffer_size * sizeof(T);
  return ret;
}

void 
raymarching_allocate_interative(RayMarchingData& params, CUDABuffer& buffer, const uint32_t numPixels, cudaStream_t stream) 
{
  const uint32_t numPixelsPadded = util::next_multiple(numPixels, 256U);

  const uint32_t nSamplesPerCoord = (params.mode == RMG) ? 4 * N_ITERS : N_ITERS;

  size_t nBytes = numPixelsPadded * nSamplesPerCoord * sizeof(vec4f); // inference input + output
  nBytes += numPixelsPadded * sizeof(uint8_t);
  nBytes += numPixelsPadded * sizeof(Packet);
  if (params.mode == RMS)  {
    nBytes += numPixelsPadded * sizeof(vec4f) * 3;
  }

  buffer.resize(nBytes, stream);
  buffer.memset(0, stream);
  char* begin = (char*)buffer.d_pointer();
  size_t offset = 0;

  // allocate staging data
  params.inference_coords = define_buffer<vec3f>(begin, offset, numPixelsPadded * nSamplesPerCoord);
  params.inference_values = define_buffer<float>(begin, offset, numPixelsPadded * nSamplesPerCoord);

  // allocate payload data 
  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    params.packets[i] = define_buffer<Packet::DataType>(begin, offset, numPixelsPadded);
  }

  // we also need a launch index buffer
  params.flags = define_buffer<uint8_t>(begin, offset, numPixelsPadded);

  // single shade payloads
  if (params.mode == RMS) { // these data are fixed output
    params.final_org_and_jitter = define_buffer<vec4f>(begin, offset, numPixelsPadded);
    params.final_highest_rgba = define_buffer<vec4f>(begin, offset, numPixelsPadded);
    params.shading_rgba = define_buffer<vec4f>(begin, offset, numPixelsPadded);
  }
  assert(offset == nBytes);
}

void
MethodRayMarching::render(cudaStream_t stream, const LaunchParams& _params, const VNRDeviceVolume& volume, const IterativeSampler& sampler, ShadingMode mode, bool iterative)
{
  TRACE_CUDA;

  RayMarchingData params = _params;
  params.volume = volume;
  params.mode = mode;
  if (iterative) {
    const uint32_t numPixels = (uint32_t)params.frame.size.long_product();
    raymarching_allocate_interative(params, sample_streaming_buffer, numPixels, stream);
    do_raymarching_iterative(stream, params, sampler, numPixels);
  }
  else {
    do_raymarching_trivial(stream, params);
  }

  TRACE_CUDA;
}


//------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<ShadingMode MODE> 
inline __device__ float
raymarching_transmittance(const VNRDeviceVolume& self,
                          const RayMarchingData& params,
                          Ray ray, RandomTEA& rng)
{
  const auto marching_step = params.raymarching_shadow_sampling_scale * self.step;
  float alpha(0);
  if (intersectVolume(ray, self)) {
    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;
    // start marching
    raymarching_iterator<ADAPTIVE_SAMPLING && MODE != RMS>(self, ray, marching_step, 
    [&] (const vec2f& t) 
    {
      // sample data value
      const auto p = ray.org + lerp(jitter, t.x, t.y) * ray.dir; // object space position
      const auto sampleValue = sampleVolume(self.volume, p);
      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
      opacityCorrection(self, t.y - t.x, sampleAlpha);
      // blending
      alpha += (1.f - alpha) * sampleAlpha;
      return alpha < nearly_one;
    });
  }
  return 1.f - alpha;
}

template<ShadingMode MODE> 
inline __device__ vec4f
raymarching_traceray(const VNRDeviceVolume& self,
                     const RayMarchingData& params,
                     const affine3f& wto, // world to object
                     const affine3f& otw, // object to world
                     RayPayload<MODE>& ray, // float t0, float t1,
                     RandomTEA& rng)
{
  const auto& marchingStep = self.step;
  const auto& gradientStep = self.grad_step;
  const auto& shadingScale = params.scivis_shading_scale;

  // vec3f gradient = 0.f;
  vec3f highestOrg   = 0.f;
  vec3f highestColor = 0.f;
  float highestAlpha = 0.f;

  float alpha(0);
  vec3f color(0);

  if (intersectVolume(ray, self)) {
    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;

    // start marching
    raymarching_iterator<ADAPTIVE_SAMPLING && MODE != RMS>(self, ray, marchingStep, 
    [&] (const vec2f& t) 
    {
      assert(t.x < t.y);

      // sample data value
      const auto p = ray.org + lerp(jitter, t.x, t.y) * ray.dir; // object space position
      const auto sampleValue = sampleVolume(self.volume, p);

      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
      opacityCorrection(self, t.y - t.x, sampleAlpha);

      // access gradient
      const vec3f No = -sampleGradient(self.volume, p, sampleValue, gradientStep); // sample gradient
      const vec3f Nw = xfmNormal(otw, No);

      const float tr = 1.f - alpha;

      // compute shading
      if constexpr (MODE == RMG) {
        const auto dir = xfmVector(otw, ray.dir);
        const vec3f shadingColor = 
          shade_scivis_light(dir, Nw, sampleColor, 
            params.mat_scivis, params.l_ambient.color, 
            params.l_distant.color, params.l_distant.direction
          );
        sampleColor = lerp(shadingScale, sampleColor, shadingColor);
      }
      else if constexpr (MODE == RMS) {
        // remember point of highest density for deferred shading
        if (highestAlpha < (1.f - alpha) * sampleAlpha) {
          highestOrg = p; // object space
          highestColor = sampleColor;
          highestAlpha = (1.f - alpha) * sampleAlpha;
        }
        // gradient += tr * Nw; // accumulate gradient for SSH
      }

      color += tr * sampleColor * sampleAlpha;
      alpha += tr * sampleAlpha;

      return alpha < nearly_one;
    });

    if (highestAlpha > 0.f) { // object space to world space
      const auto ldir = xfmVector(wto, normalize(params.l_distant.direction));
      const auto rdir = xfmVector(otw, ray.dir);
      const float transmittance = raymarching_transmittance<MODE>(
        self, params, Ray{highestOrg, ldir, 0.f, float_large}, rng
      );
      color = lerp(shadingScale, color, highestColor * alpha * transmittance);
    }
  }

  return vec4f(color, alpha);
}

template<ShadingMode MODE> 
__global__ void
raymarching_kernel(uint32_t width, uint32_t height, const RayMarchingData params)
{
  // compute pixel ID
  const size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width)  return;
  if (iy >= height) return;

  const auto& volume = params.volume;
  assert(width  == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  // pixel index
  const uint32_t fbIndex = ix + iy * width;

  // random number generator
  RandomTEA rng_state(params.frame_index, fbIndex);

  // generate ray direction
  RayPayload<MODE> ray;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));

  // trace ray
  const vec4f output = raymarching_traceray(volume, params, wto, otw, ray, rng_state);

  // and write to frame buffer ...
  writePixelColor(params, output, fbIndex);
}

void
do_raymarching_trivial(cudaStream_t stream, const RayMarchingData& params)
{
  if (params.mode == RMB) {
    util::bilinear_kernel(raymarching_kernel<RMB>, 0, stream, params.frame.size.x, params.frame.size.y, params);
  }
  else if (params.mode == RMG) {
    util::bilinear_kernel(raymarching_kernel<RMG>, 0, stream, params.frame.size.x, params.frame.size.y, params);
  }
  else if (params.mode == RMS) {
    util::bilinear_kernel(raymarching_kernel<RMS>, 0, stream, params.frame.size.x, params.frame.size.y, params);
  }
  else {
    throw std::runtime_error("unknown shading mode");
  }
}



// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<ShadingMode MODE> 
__global__ void
iterative_intersect_kernel(uint32_t numRays, const RayMarchingData params, int N_ITERS)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = params.volume;
  const auto& gradientStep = self.grad_step;
  MAYBE_UNUSED const auto& volume = self.volume;

  RayPayload ray = load<MODE>(params, i);

  vec3f* __restrict__ coords = (vec3f*)params.inference_coords;

  int k = 0;
  ray.iter.exec(self, ray, self.step, ray.pidx, [&] (const vec2f& t, const float r) {
    assert(k < N_ITERS);
    assert(t.x < t.y);

    // object space position
    const vec3f c = ray.org + lerp(ray.jitter, t.x, t.y) * ray.dir;
    // const vec3f c = ray.org + t.x * ray.dir;
    coords[numRays * k + i] = c;

    // object space gradient
    if (MODE == RMG) {
      const vec3f gx = c + vec3f(gradientStep.x, 0, 0);
      const vec3f gy = c + vec3f(0, gradientStep.y, 0);
      const vec3f gz = c + vec3f(0, 0, gradientStep.z);
      coords[1 * numRays * N_ITERS + (numRays * k + i)] = gx; 
      coords[2 * numRays * N_ITERS + (numRays * k + i)] = gy; 
      coords[3 * numRays * N_ITERS + (numRays * k + i)] = gz; 
    }

    return (++k) < N_ITERS;
  });
}

template<ShadingMode MODE> 
__global__ void
iterative_compose_kernel(uint32_t numRays, const RayMarchingData params, int N_ITERS)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = params.volume;
  const auto& otw = params.volume.transform;
  const auto& gradientStep = self.grad_step;
  const auto& shadingScale = params.scivis_shading_scale;
  const auto* __restrict__ samples = params.inference_values;

  RayPayload ray = load<MODE>(params, i);

  int k = 0;
  ray.iter.exec(self, ray, self.step, ray.pidx, [&] (const vec2f& t, const float r) {
    assert(k < N_ITERS);
    assert(t.x < t.y);

    // classification
    const auto c = ray.org + lerp(ray.jitter, t.x, t.y) * ray.dir;
    const auto sampleValue = samples[numRays * k + i];
    vec3f sampleColor;
    float sampleAlpha;
    sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
    opacityCorrection(self, t.y - t.x, sampleAlpha);

    // shading
    if constexpr (MODE == RMG) {
      // compute sample gradient
      const auto fgx = samples[1 * numRays * N_ITERS + numRays * k + i];
      const auto fgy = samples[2 * numRays * N_ITERS + numRays * k + i];
      const auto fgz = samples[3 * numRays * N_ITERS + numRays * k + i];
      const vec3f No = -vec3f(fgx - sampleValue, fgy - sampleValue, fgz - sampleValue) / gradientStep;
      const vec3f Nw = xfmNormal(otw, No);
      // calculate lighting in the world space 
      const auto dir = xfmVector(otw, ray.dir);
      const vec3f shadingColor = 
        shade_scivis_light(dir, Nw, sampleColor, 
          params.mat_scivis, params.l_ambient.color, 
          params.l_distant.color, params.l_distant.direction
        );
      sampleColor = lerp(shadingScale, sampleColor, shadingColor);
    }
    else if constexpr (MODE == RMS) {
      if (ray.ss.highest_alpha < (1.f - ray.alpha) * sampleAlpha) {
        ray.ss.highest_org = c;
        ray.ss.highest_color = sampleColor;
        ray.ss.highest_alpha = (1.f - ray.alpha) * sampleAlpha;
      }
    }
    
    // blending
    const float tr = 1.f - ray.alpha;
    ray.alpha += tr * sampleAlpha;
    if constexpr (MODE != SHADOW) {
      ray.color += tr * sampleColor * sampleAlpha;
    }

    // conditions to continue iterating
    return ((++k) < N_ITERS) && (ray.alpha < nearly_one);
  });

  const bool resumable = ray.iter.resumable(self, ray);
  if (ray.alpha < nearly_one && resumable) {
    save<MODE>(params, ray, i);
  }
  else {
    const uint32_t& pidx = ray.pidx;
    if constexpr (MODE == SHADOW) {
      vec4f shadingColor = params.shading_rgba[pidx];
      vec4f highestColor = params.final_highest_rgba[pidx];
      shadingColor.xyz() = lerp(shadingScale, 
        shadingColor, highestColor * shadingColor.w * (1.f - ray.alpha)
      ).xyz();
      writePixelColor(params, shadingColor, pidx);
    }
    else if constexpr (MODE == RMS) {
      params.final_org_and_jitter[i].xyz() = ray.ss.highest_org;
      params.final_highest_rgba[pidx] = vec4f(ray.ss.highest_color, ray.ss.highest_alpha);
      params.shading_rgba[pidx] = vec4f(ray.color, ray.alpha);
    }
    else {
      writePixelColor(params, vec4f(ray.color, ray.alpha), pidx);
    }
  }
}

__global__ void
iterative_raygen_kernel_camera(uint32_t numRays, const RayMarchingData params) 
{
  // compute ray ID
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  // generate data
  const auto& self = params.volume;

  // random number generator
  RandomTEA rng = RandomTEA(params.frame_index, i);
  vec2f jitters = rng.get_floats();

  // payload & ray
  RayPayload<RMB> ray;  
  ray.pidx = i;
  ray.jitter = jitters.x;
  compute_ray<RMB>(params, ray);

  if (params.mode == RMS) {
    params.final_org_and_jitter[i].w = jitters.y;
  }

  // intersect with volume bbox & write outputs
  if (intersectVolume(ray, self)) {
    ray.iter = Iter(self, ray);
    save<RMB>(params, ray, i);
  }
  else {
    if (params.mode != RMS) {
      writePixelColor(params, vec4f(ray.color, ray.alpha), ray.pidx);
    }
  }
}

__global__ void
iterative_raygen_kernel_shadow(uint32_t numRays, const RayMarchingData params) 
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = params.volume;

  vec4f org_and_jitter = params.final_org_and_jitter[i];

  RayPayload<SHADOW> ray;  
  ray.pidx = i;
  ray.jitter = org_and_jitter.w;
  ray.org = org_and_jitter.xyz();
  compute_ray<SHADOW>(params, ray);

  if (intersectVolume(ray, self) 
    && (params.final_highest_rgba[i].w > 0.f)) 
  {
    ray.iter = Iter(self, ray);
    save<SHADOW>(params, ray, i); 
  }
  else {
    writePixelColor(params, params.shading_rgba[i], i);
  }
}

inline bool 
iterative_ray_compaction(cudaStream_t stream, uint32_t& numRays, const RayMarchingData& params)
{
  return inplace_compaction<IterData, IterView>(stream, numRays, params.flags, IterView(params, 0)) > 0;
}

template<ShadingMode MODE> 
void iterative_raymarching_loop(cudaStream_t stream, const RayMarchingData& params, const IterativeSampler& sampler, uint32_t numRays)
{
  const uint32_t numCoordsPerSample = (MODE == RMG) ? 4 * N_ITERS : N_ITERS;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (MODE == SHADOW) {
    util::linear_kernel(iterative_raygen_kernel_shadow, 0, stream, numRays, params);
  }
  else {
    util::linear_kernel(iterative_raygen_kernel_camera, 0, stream, numRays, params);
  }
  TRACE_CUDA;
  while (iterative_ray_compaction(stream, numRays, params)) {
    // Actually, we could have merged the intersection step with raygen and compose. However, there was a wired error 
    // and I did not figure out irs origin. Also, having the intersection step inside raygen and compose did not bring
    // obvious performance benefit, so I left it as it is for now.
    TRACE_CUDA;
    util::linear_kernel(iterative_intersect_kernel<MODE>, 0, stream, numRays, params, N_ITERS);
    TRACE_CUDA;
    sampler(stream, numCoordsPerSample * numRays, params.inference_coords, params.inference_values);
    TRACE_CUDA;
    util::linear_kernel(iterative_compose_kernel<MODE>, 0, stream, numRays, params, N_ITERS);
    TRACE_CUDA;
  }
}

void
do_raymarching_iterative(cudaStream_t stream, const RayMarchingData& params, const IterativeSampler& sampler, uint32_t numRays)
{
  if (params.mode == RMB) {
    iterative_raymarching_loop<RMB>(stream, params, sampler, numRays);
  }
  else if (params.mode == RMG) {
    iterative_raymarching_loop<RMG>(stream, params, sampler, numRays);
  }
  else if (params.mode == RMS) {
    iterative_raymarching_loop<RMS>(stream, params, sampler, numRays);
    iterative_raymarching_loop<SHADOW>(stream, params, sampler, numRays);
  }
}

} // namespace vnr
