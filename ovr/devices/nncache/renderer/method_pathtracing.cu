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

#ifndef ADAPTIVE_SAMPLING
#define ADAPTIVE_SAMPLING 1
#endif
// #define ADAPTIVE_SAMPLING 0

namespace ovr::nncache {

// NOTE: use 6 for comparison with openvkl, otherwise use 2
// #define max_num_scatters 16
static constexpr int russian_roulette_length = 4;

static inline vec3f __device__ PHASE(const vec3f& albedo) { 
  return albedo * 0.6f; 
}

using Rng = random::RandomTEA;

struct PathTracingPackedRay;
struct PathTracingRay;
using PackedRay = PathTracingPackedRay;
using Ray = PathTracingRay;

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<bool ADAPTIVE>
struct PathTracingIter;

template<>
struct PathTracingIter<true> : private dda::DDAIter {
public:
  using DDAIter::cell;
  using DDAIter::t_next;
  using DDAIter::next_cell_begin;
public:
  __device__ PathTracingIter();
  __device__ PathTracingIter(const VNRDeviceVolume& self, const Ray& ray);
  __device__ bool hashit(const VNRDeviceVolume& self, const Ray& ray, float& rayt, float& majorant);
  __device__ bool finished(const VNRDeviceVolume& self, const Ray& ray);
};

template<>
struct PathTracingIter<false> {
public:
  float t;
private:
  uint32_t __pad[2];
public:
  __device__ PathTracingIter();
  __device__ PathTracingIter(const VNRDeviceVolume& self, const Ray& ray);
  __device__ bool hashit(const VNRDeviceVolume& self, const Ray& ray, float& rayt, float& majorant);
  __device__ bool finished(const VNRDeviceVolume& self, const Ray& ray);
};

using Iter = PathTracingIter<ADAPTIVE_SAMPLING>;


// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<bool ADAPTIVE>
__device__ bool
delta_tracking(const VNRDeviceVolume& self, /* object space */ const Ray& ray, float& _t, vec3f& _albedo);

template<>
__device__ bool 
delta_tracking<true>(const VNRDeviceVolume& self, /* object space */ const Ray& ray,float& _t, vec3f& _albedo);

template<>
__device__ bool 
delta_tracking<false>(const VNRDeviceVolume& self, /* object space */ const Ray& ray, float& _t, vec3f& _albedo);


// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

struct PathTracingPackedRay {
public:
  // 0 //
  uint32_t pidx : 24;
  uint32_t shadow : 8;
  vec3f org{};
  // 16 //
  vec3f dir{};
  uint32_t scatter_index{};
  // 32 //
  float majorant{};
  vec3f L = vec3f(0.0);
  // 48 //
  vec3f throughput = vec3f(1.0);
  mutable Rng rng;
  // 64 //
  Iter iter;
  uint32_t __pad;
public:
  __device__ PathTracingPackedRay() : pidx(0), shadow(0) {};
};

struct PathTracingRay : PathTracingPackedRay {
  float tnear{};
  float tfar{};
  vec3f sample_coord{};
  float sample_value{};
};

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

using Packet = PacketTemplate<PackedRay>;
static_assert((sizeof(PackedRay)) % sizeof(uint4) == 0, "Incorrect size of PackedRay");

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

#include "pathtracing_iter.inc"

struct PathTracingData : LaunchParams
{
  PathTracingData(const LaunchParams& p) : LaunchParams(p) {}

  VNRDeviceVolume volume;

  vec3f* inference_coords{ nullptr };
  float* inference_values{ nullptr };
  Packet::DataType* packets[Packet::N] = {};
  uint8_t* flags{ nullptr };
};

struct IterData {
  Packet packet;
  vec3f coord;
  float value;
};

struct IterView {
  Packet::DataType* packets[Packet::N];
  vec3f* coords;
  float* values;

  __host__ __device__ IterView() {}

  __host__ IterView(const PathTracingData& params, uint32_t offset) 
    : coords(params.inference_coords + offset)
    , values(params.inference_values + offset)
  {
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
    coords += n; values += n;
  }

  __host__ __device__ ptrdiff_t diff(const IterView& other) const {
    return values - other.values;
  }

  /// Conversion operator
  __host__ __device__ operator IterData() const {
    IterData value;
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      value.packet.data[i] = *this->packets[i];
    }
    value.coord = *coords;
    value.value = *values;
    return value;
  }

  /// Assignment operator
  __host__ __device__ IterView& operator=(const IterData& value) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      *this->packets[i] = value.packet.data[i];
    }
    *coords = value.coord;
    *values = value.value;
    return *this;
  }
};

inline __device__ Ray 
load(const PathTracingData& params, const uint32_t& ridx) 
{
  Packet packet;

  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    packet.data[i] = params.packets[i][ridx];
  }

  Ray ray = (const Ray&)packet.items;
  ray.sample_coord = params.inference_coords[ridx];
  ray.sample_value = params.inference_values[ridx];
  ray.tnear = 0.f;
  ray.tfar = float_large;  

  // verify the loaded ray data
  const auto& self = params.volume;
  const bool valid = intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self);

  // keep this here for a while
  assert(valid && "[load] invalid ray being loaded");
  // if (!valid)  {
  //   printf("[error pt] invalid ray: ridx = %d, t = (%f, %f), org = (%f, %f, %f), dir = (%f, %f, %f)\n", 
  //     (int)ridx, ray.tnear, ray.tfar, 
  //     ray.org.x, ray.org.y, ray.org.z, 
  //     ray.dir.x, ray.dir.y, ray.dir.z);
  // }
  // return valid;

  return ray;
}

inline __device__ void 
save(const PathTracingData& params, const Ray& ray, const uint32_t& ridx)
{
  assert(ray.tnear < ray.tfar && "[save] invalid ray being saved");  
  Packet packet = PackedRay(ray);
  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    params.packets[i][ridx] = packet.data[i];
  }
  params.inference_coords[ridx] = ray.sample_coord;
  params.inference_values[ridx] = ray.sample_value;
  params.flags[ridx] = 0xFF;
}

/* volume decoding version */ void
do_path_tracing_trivial(cudaStream_t stream, const PathTracingData& params);

/* sample streaming version */ void
do_path_tracing_iterative(cudaStream_t stream, const PathTracingData& params, const IterativeSampler& sampler, uint32_t numRays);


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T>
inline T* 
define_buffer(char*& curr, size_t buffer_size)
{
  auto* ret = (T*)(curr); 
  curr = curr + buffer_size * sizeof(T);
  return ret;
}

void 
pathtracing_allocate_interative(PathTracingData& params, 
  CUDABuffer& fbuffer, CUDABuffer& pbuffer, CUDABuffer& sbuffer, 
  const uint32_t numPixels, cudaStream_t stream) 
{
  const uint32_t numPixelsPadded = util::next_multiple(numPixels, 256U);

  // allocate space for the ray flags
  fbuffer.resize(numPixelsPadded * sizeof(uint8_t), stream);
  fbuffer.memset(0, stream);
  params.flags = (uint8_t*)fbuffer.d_pointer();

  // allocate space for packets
  pbuffer.resize(numPixelsPadded * sizeof(Packet), stream);
  {
    char* curr = (char*)pbuffer.d_pointer();
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      params.packets[i] = define_buffer<Packet::DataType>(curr, numPixelsPadded);
    }
    assert(curr - (char*)pbuffer.d_pointer() == pbuffer.sizeInBytes
           && "incorrect buffer allocation");
    // params.packets_data = (Packet::DataType*)pbuffer.d_pointer();
  }

  // allocate space for sample streaming data
  sbuffer.resize(numPixelsPadded * (sizeof(vec3f) + sizeof(float)), stream); // coords + values
  {
    auto* curr = (char*)sbuffer.d_pointer();
    params.inference_coords = define_buffer<vec3f>(curr, numPixelsPadded);
    params.inference_values = define_buffer<float>(curr, numPixelsPadded);
    assert(curr - (char*)sbuffer.d_pointer() == sbuffer.sizeInBytes 
           && "incorrect buffer allocation");
  }
}

void
MethodPathTracing::render(cudaStream_t stream, const LaunchParams& _params, const VNRDeviceVolume& volume, const IterativeSampler& sampler, bool iterative)
{
  PathTracingData params = _params;
  params.volume = volume;
  if (iterative) {
    const uint32_t numPixels = (uint32_t)params.frame.size.long_product();
    pathtracing_allocate_interative(params, counter_buffer, packets_buffer, samples_buffer, numPixels, stream);
    do_path_tracing_iterative(stream, params, sampler, numPixels);
  }
  else {
    do_path_tracing_trivial(stream, params);
  }
}


//------------------------------------------------------------------------------
// do_path_tracing_trivial
// ------------------------------------------------------------------------------

inline __device__ float 
luminance(const vec3f &c)
{
  return 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
}

inline __device__ bool 
russian_roulette(vec3f& throughput, Rng& rng, const int32_t& scatter_index)
{
  if (scatter_index > russian_roulette_length) { 
    float q = std::min(0.95f, /*luminance=*/reduce_max(throughput));
    if (rng.get_float() > q) {
      return true;
    }
    throughput /= q;
  }
  return false;
}

inline __device__ vec3f
path_tracing_reference(const PathTracingData& params, const affine3f& wto, Ray ray)
{
  const auto& self = params.volume;

  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  float t;
  vec3f albedo;

  int scatter_index = 0;
  while (intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) {

    // ray exits the volume, compute lighting
    if (!delta_tracking<ADAPTIVE_SAMPLING>(self, ray, t, albedo)) {
      if (scatter_index > 0) { // no light accumulation for primary rays
        L += throughput * params.l_ambient.color;
      }
      break;
    }

    // terminate ray
    if (russian_roulette(throughput, ray.rng, scatter_index))
      break;
    ++scatter_index;

    // reset ray
    ray.org = ray.org + t * ray.dir;
    ray.tnear = 0.f;
    ray.tfar = float_large;
    throughput *= PHASE(albedo);

    // direct lighting
    ray.dir = xfmVector(wto, normalize(params.l_distant.direction));
    if (!delta_tracking<ADAPTIVE_SAMPLING>(self, ray, t, albedo)) {
      L += throughput * params.l_distant.color;
    }

    // scattering
    ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, ray.rng.get_floats()));
  }

  return L;
}

inline __device__ vec3f
path_tracing_traceray(const PathTracingData& params, const affine3f& wto, Ray ray/* object space ray */)
{
  const auto& self = params.volume;

  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  float t;
  vec3f albedo;

  int scatter_index = 0;
  while (intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) {
    const bool exited = !delta_tracking<ADAPTIVE_SAMPLING>(self, ray, t, albedo);

    if (ray.shadow) {

      if (exited) {
        L += throughput * params.l_distant.color;
      }

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, ray.rng.get_floats()));
      ray.shadow = 0;
    }
    else {

      if (exited) {              // ray exits the volume, compute lighting
        if (scatter_index > 0) { // no light accumulation for primary rays
          L += throughput * params.l_ambient.color;
        }
        break;
      }

      if (russian_roulette(throughput, ray.rng, scatter_index)) break;
      ++scatter_index;

      ray.org = ray.org + t * ray.dir;
      throughput *= PHASE(albedo);

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(wto, normalize(params.l_distant.direction));
      ray.shadow = 1;
    }
  }

  return L;
}

__global__ void
path_tracing_kernel(uint32_t width, uint32_t height, const PathTracingData params)
{
  // compute pixel ID
  const size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width) return;
  if (iy >= height) return;

  assert(width == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  const uint32_t pidx = ix + iy * width; // pixel index

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  // generate ray & payload
  Rng rng(params.frame_index, pidx);

  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;
  ray.rng = Rng(params.frame_index, pidx);

  // trace ray
  const vec3f color = path_tracing_traceray(params, wto, ray);

  // and write to frame buffer ... (accumilative)
  writePixelColor(params, vec4f(color, 1.f), pidx);
}

void
do_path_tracing_trivial(cudaStream_t stream, const PathTracingData& params)
{
  util::bilinear_kernel(path_tracing_kernel, 0, stream, params.frame.size.x, params.frame.size.y, params);
}



// ------------------------------------------------------------------------------
// do_path_tracing_iterative
// ------------------------------------------------------------------------------

inline __device__ bool
iterative_take_sample(const PathTracingData& params, Ray& ray)
{
  const auto& self = params.volume;
  const affine3f otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  float t;
  if (ray.iter.hashit(self, ray, t, ray.majorant)) {
    ray.sample_coord = ray.org + t * ray.dir; // object space position  
    return true;
  }

  // ray exits the volume, compute lighting
  assert(ray.iter.finished(self, ray));
  if (ray.scatter_index > 0) { // no light accumulation for primary rays
    if (ray.shadow) {
      ray.L += ray.throughput * params.l_distant.color;
      
      ray.shadow = 0;
      ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, ray.rng.get_floats()));
      
      if (!intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) return false;
      ray.iter = Iter(self, ray);

      if (ray.iter.hashit(self, ray, t, ray.majorant)) {
        ray.sample_coord = ray.org + t * ray.dir; // object space position  
        return true;
      }
    }
    else {
      ray.L += ray.throughput * params.l_ambient.color;
    }
  }
  return false;
}

inline __device__ bool
iterative_shade(const PathTracingData& params, Ray& ray)
{
  const auto& self = params.volume;
  const affine3f otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  // handle collision
  const auto rgba = sampleTransferFunction(self.tfn, ray.sample_value);
  if (ray.rng.get_float() * ray.majorant >= rgba.w * self.density_scale) return true;

  const vec3f albedo = rgba.xyz();

  if (ray.shadow) {
    ray.shadow = 0;
    ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, ray.rng.get_floats()));
  }
  else {
    if (russian_roulette(ray.throughput, ray.rng, ray.scatter_index)) return false;
    ++ray.scatter_index;

    ray.org = ray.sample_coord;
    ray.tnear = 0.f;
    ray.tfar = float_large;
    ray.throughput *= PHASE(albedo);

    ray.shadow = 1;
    ray.dir = xfmVector(wto, normalize(params.l_distant.direction));
  }

  if (!intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) return false;
  ray.iter = Iter(self, ray);

  return true;
}

__global__ void
iterative_raygen_kernel(uint32_t numRays, const PathTracingData params) 
{
  // compute ray ID
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;
  const uint32_t pidx = i; // pixel index
  const uint32_t ix = pidx % params.frame.size.x;
  const uint32_t iy = pidx / params.frame.size.x;

  // generate data
  const auto& self = params.volume;

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  // generate ray
  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;
  ray.rng = Rng(params.frame_index, pidx);

  // initialize rays
  if (intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) {
    ray.iter = Iter(self, ray);
    if (iterative_take_sample(params, ray)) {
      save(params, ray, i); return;
    }
  }

  writePixelColor(params, vec4f(ray.L, 1.f), ray.pidx);
}

__global__ void
iterative_shade_kernel(uint32_t numRays, const PathTracingData params) 
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; // compute ray ID
  if (i >= numRays) return;

  MAYBE_UNUSED const auto& self = params.volume;
  Ray ray = load(params, i);

  if (iterative_shade(params, ray) && iterative_take_sample(params, ray)
   && intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) 
  {
    save(params, ray, i);
  }
  else {
    writePixelColor(params, vec4f(ray.L, 1.f), ray.pidx);
  }
}

inline bool 
iterative_ray_compaction(cudaStream_t stream, uint32_t& numRays, const PathTracingData& params)
{
  return inplace_compaction<IterData, IterView>(stream, numRays, params.flags, IterView(params, 0)) > 0;
}

void 
do_path_tracing_iterative(cudaStream_t stream, const PathTracingData& params, const IterativeSampler& sampler, uint32_t numRays)
{
  CUDA_CHECK(cudaStreamSynchronize(stream));
  util::linear_kernel(iterative_raygen_kernel, 0, stream, numRays, params);
  while (iterative_ray_compaction(stream, numRays, params)) {
    sampler(stream, numRays, params.inference_coords, params.inference_values);
    util::linear_kernel(iterative_shade_kernel, 0, stream, numRays, params);
  }
}

} // namespace vnr
