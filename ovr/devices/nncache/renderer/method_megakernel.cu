#include "methods.h"
#include "dda.h"
#include "../cachemanager.h"

#include <random/random.h>

#include <iostream>
#include <iomanip> // std::setw

#define float_epsilon std::numeric_limits<float>::epsilon()
#define float_large 1e31f
#define float_small 1e-31f
#define nearly_one 0.9999f

// NOTE: use 6 for comparison with openvkl, otherwise use 2
// #define max_num_scatters 16
#define russian_roulette_length 4

#define ray_marching_shading_scale 0.9f
#define ray_marching_shadow_sampling_scale 2.f

#define ENABLE_DATA_CACHING

namespace ovr {

using misc::bilinear_kernel;
using random::RandomTEA;
using tdns::gpucache::K_CacheManager;

namespace nncache {

template<typename T>
struct DeviceVolume : DeviceGrid {
public:
  mutable tdns::gpucache::K_CacheManager<T> cache;
  // NeuralVolume* neuralrepr { nullptr };
  // TODO: add more data here
  uint32_t max_lod;

public:
  DeviceVolume(DeviceGrid& grid, tdns::gpucache::K_CacheManager<T> cache) : DeviceGrid(grid), cache(cache) {}
};

struct Ray {
  vec3f org{};
  vec3f dir{};
  float tnear{};
  float tfar{};

  uint32_t pidx{0};

  bool shadow{false};

  const affine3f* wto{};
  const affine3f* otw{};
  random::RandomTEA* rng{};
};

// ------------------------------------------------------------------
// ray tracing helper
// ------------------------------------------------------------------

static __device__ bool
intersect_box(float& _t0, float& _t1, const vec3f ray_ori, const vec3f ray_dir, const box3f& box)
{
  const vec3f& lower = box.lower;
  const vec3f& upper = box.upper;

  float t0 = _t0;
  float t1 = _t1;
#if 1
  const vec3i is_small =
    vec3i(fabs(ray_dir.x) < float_small, fabs(ray_dir.y) < float_small, fabs(ray_dir.z) < float_small);
  const vec3f rcp_dir = /* ray direction reciprocal*/ 1.f / ray_dir;
  const vec3f t_lo = vec3f(is_small.x ? float_large : (lower.x - ray_ori.x) * rcp_dir.x, //
                           is_small.y ? float_large : (lower.y - ray_ori.y) * rcp_dir.y, //
                           is_small.z ? float_large : (lower.z - ray_ori.z) * rcp_dir.z  //
  );
  const vec3f t_hi = vec3f(is_small.x ? -float_large : (upper.x - ray_ori.x) * rcp_dir.x, //
                           is_small.y ? -float_large : (upper.y - ray_ori.y) * rcp_dir.y, //
                           is_small.z ? -float_large : (upper.z - ray_ori.z) * rcp_dir.z  //
  );
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#else
  const vec3f t_lo = (lower - ray_ori) / ray_dir;
  const vec3f t_hi = (upper - ray_ori) / ray_dir;
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#endif
  _t0 = t0;
  _t1 = t1;
  return t1 > t0;
}

template<typename T, int N>
static __device__ T
array1d_nodal(const ArrayCUDA<1, N>& array, float v)
{
  float t = (0.5f + v * (array.dims.v - 1)) / array.dims.v;
  return tex1D<T>(array.data, t);
}

static __device__ float MAYBE_UNUSED 
sample_volume(const Array3DScalarCUDA& self, vec3f p)
{
  /* sample volume in object space [0, 1] */
  p.x = clamp(p.x, 0.f, 1.f);
  p.y = clamp(p.y, 0.f, 1.f);
  return tex3D<float>(self.data, p.x, p.y, p.z);
}

static __device__ vec3f MAYBE_UNUSED 
sample_gradient(const Array3DScalarCUDA& self,
                const vec3f c, // central position
                const float v, // central value
                vec3f stp)
{
  vec3f ext = c + stp;
  if (ext.x > 1.f)
    stp.x *= -1.f;
  if (ext.y > 1.f)
    stp.y *= -1.f;
  if (ext.z > 1.f)
    stp.z *= -1.f;
  const vec3f gradient(sample_volume(self, c + vec3f(stp.x, 0, 0)) - v,  //
                       sample_volume(self, c + vec3f(0, stp.y, 0)) - v,  //
                       sample_volume(self, c + vec3f(0, 0, stp.z)) - v); //
  return normalize(gradient / stp);
}

#ifdef ENABLE_DATA_CACHING

template<typename T>
static __device__ bool
sample_volume(const DeviceVolume<T>& self, vec3f p, float* out, uint32_t lod)
{
  tdns::gpucache::VoxelStatus voxelStatus = self.cache.template get_normalized<float>(lod, p, *out);
  // handle unmapped and empty bricks
  if (/* empty space */voxelStatus == tdns::gpucache::VoxelStatus::Empty || 
      /* cache miss  */voxelStatus == tdns::gpucache::VoxelStatus::Unmapped) 
  {
    return false; // advance to next sampling position
  }
  return true;
}

template<typename T>
static __device__ bool
sample_volume_gradient(const DeviceVolume<T>& self, const vec3f c, /* central position */
                       float* value, vec3f* grad, uint32_t lod)
{
  auto stp = self.grad_step;
  vec3f ext = c + stp;
  if (ext.x > 1.f) stp.x *= -1.f;
  if (ext.y > 1.f) stp.y *= -1.f;
  if (ext.z > 1.f) stp.z *= -1.f;
  if (!sample_volume(self, c, value, lod)) return false;
  float x, y, z;
  bool x_ok = sample_volume(self, c + vec3f(stp.x, 0, 0), &x, lod);
  bool y_ok = sample_volume(self, c + vec3f(0, stp.y, 0), &y, lod);
  bool z_ok = sample_volume(self, c + vec3f(0, 0, stp.z), &z, lod);
  if (x_ok && y_ok && z_ok) {
    *grad = normalize((vec3f(x, y, z) - *value) / stp);
    return true;
  }
  return false;
}

#endif

static __device__ void
sample_transfer_function(const DeviceTransferFunction& tfn, float sampleValue, vec3f& _sampleColor, float& _sampleAlpha)
{
  const auto v = (clamp(sampleValue, tfn.value_range.lo, tfn.value_range.hi) - tfn.value_range.lo) * tfn.range_rcp_norm;
  vec4f rgba = array1d_nodal<float4>(tfn.colors, v);
  rgba.w = array1d_nodal<float>(tfn.alphas, v); // followed by the alpha correction
  _sampleColor = vec3f(rgba);
  _sampleAlpha = rgba.w;
}

static __device__ vec4f // it looks like we can only read textures as float
sample_transfer_function(const Array1DFloat4CUDA& texColors,
                         const Array1DScalarCUDA& texAlphas,
                         const float lower,
                         const float upper,
                         const float scale,
                         float sampleValue)
{
  const auto v = (clamp(sampleValue, lower, upper) - lower) * scale;
  vec3f sampleColor = vec3f(array1d_nodal<float4>(texColors, v));
  float sampleAlpha = array1d_nodal<float>(texAlphas, v); // followed by the alpha correction
  return vec4f(sampleColor, sampleAlpha);
}

static __device__ void
opacity_correction(const DeviceGrid& self, const float& distance, float& opacity)
{
  opacity = 1.f - __powf(1.f - opacity, self.step_rcp * distance);
}

static __device__ vec3f
cartesian(const float phi, const float sinTheta, const float cosTheta)
{
  float sinPhi, cosPhi;
  sincosf(phi, &sinPhi, &cosPhi);
  return vec3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

static __device__ vec3f
uniform_sample_sphere(const float radius, const vec2f s)
{
  const float phi = 2 * M_PI * s.x;
  const float cosTheta = radius * (1.f - 2.f * s.y);
  const float sinTheta = 2.f * radius * sqrt(s.y * (1.f - s.y));
  return cartesian(phi, sinTheta, cosTheta);
}

template <typename T>
__forceinline__ __device__ T lerp(float r, const T& a, const T& b) 
{
  return (1-r) * a + r * b;
}

inline __device__ void
write_pixel(const LaunchParams& params, const vec4f& color, const uint32_t pixel_index)
{
  vec4f rgba = color;
  if (params.frame_index == 1) {
    params.accumulation[pixel_index] = rgba;
  } else {
    rgba = params.accumulation[pixel_index] + rgba;
    params.accumulation[pixel_index] = rgba;
  }
  params.frame.rgba[pixel_index] = rgba / (float)params.frame_index;
}

// ------------------------------------------------------------------
// dda iter helper
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Helper Shading Functions
// Note: all inputs are world space vectors
// ------------------------------------------------------------------

inline __device__ vec3f
shade_simple_light(const vec3f& ray_dir, const vec3f& normal, const vec3f& albedo)
{
  if (dot(normal, normal) > 1.0e-6) {
    return albedo * (0.2f + .8f * fabsf(dot(-ray_dir, normalize(normal))));
  }
  return 0.f;
}

inline __device__ vec3f
shade_scivis_light(const vec3f& ray_dir, const vec3f& normal, const vec3f& albedo, const PhongMaterial& mat,
                   const vec3f& light_ambient, const vec3f& light_diffuse, const vec3f& light_dir)
{
  vec3f color = 0.f;

  if (dot(normal, normal) > 1.0e-6) {
    const auto L = normalize(light_dir);
    const auto N = normalize(normal);
    const auto V = -ray_dir;
    color += mat.ambient * albedo;
    const float cosNL = std::max(dot(N, L), 0.f);
    if (cosNL > 0.0f) {
      color += mat.diffuse * cosNL * albedo * light_diffuse;
      const vec3f H = normalize(L + V);
      const float cosNH = std::max(dot(N, H), 0.f);
      color += mat.specular * powf(cosNH, mat.shininess) * light_diffuse;
    }
  }

  return shade_simple_light(ray_dir, normal, color);
}

inline __device__ float
adaptive_sampling_rate(float base_sampling_step, float max_opacity)
{
  const float scale = 15 * base_sampling_step;
  const float r = fabsf(clamp(max_opacity, 0.1f, 1.f) - 1.f);
  return max(base_sampling_step + scale * std::pow(r, 2.f), base_sampling_step);
}

inline __device__ float
sample_size_scaler(const float ss, const float t0, const float t1) {
  const int32_t N = (t1-t0) / ss + 1;
  return (t1-t0) / N;
}

template<typename T, typename F, bool ADAPTIVE = true>
inline __device__ void
ray_marching_iterator(const DeviceVolume<T>& self, 
                      const vec3f& org, const vec3f& dir,
                      const float tMin, const float tMax, 
                      const float step, const F& body, 
                      bool debug = false)
{
  if (ADAPTIVE) {

    const auto& dims = self.sp.dims;
    const vec3f m_org = org / float(SingleSpacePartiton::Device::MACROCELL_SIZE);
    const vec3f m_dir = dir / float(SingleSpacePartiton::Device::MACROCELL_SIZE);

    dda::dda3(m_org, m_dir, tMin, tMax, dims, debug, [&](const vec3i& cell, float t0, float t1) {
      // calculate max opacity
      float r = self.sp.access_majorant(cell);
      if (fabsf(r) <= float_epsilon) return true; // the cell is empty
      // estimate a step size
      const auto ss = sample_size_scaler(adaptive_sampling_rate(step, r), t0, t1);
      // iterate within the interval
      vec2f t = vec2f(t0, min(t1, t0 + ss));
      while (t.y > t.x) {
        if (!body(t)) return false;
          t.x = t.y;
          t.y = min(t.x + ss, t1);
      }
      return true;
    });

  } else {

    vec2f t = vec2f(tMin, min(tMax, tMin + step));
    while ((t.y > t.x) && body(t)) {
      t.x = t.y;
      t.y = min(t.x + step, tMax);
    }

  }
}

template<typename T>
inline __device__ float
ray_marching_transmittance(const DeviceVolume<T>& self,
                           const LaunchParams& params,
                           Ray& ray, // object space ray
                           uint32_t lod)
{
  const auto marching_step = ray_marching_shadow_sampling_scale * self.step;
  const auto& org = ray.org;
  const auto& dir = ray.dir;
  const auto& otw = *ray.otw;
  auto t0 = ray.tnear;
  auto t1 = ray.tfar;
  auto& rng = *ray.rng;
  float alpha(0);
  if (intersect_box(t0, t1, org, dir, self.bbox)) {
    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;
    // start marching
    ray_marching_iterator(self, xfmPoint(otw, org), xfmNormal(otw, dir), t0, t1, marching_step, [&](const vec2f& t) {
      // sample data value
      const float tt = lerp(jitter, t.x, t.y);
      const auto p = org + tt * dir; // object space position
#ifdef ENABLE_DATA_CACHING
      float sampleValue = 0.f;
      if (!sample_volume(self, p, &sampleValue, lod)) {
        return true; // advance t to next sampling position
      }
#else
      const auto sampleValue = sample_volume(self.volume, p);
#endif
      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sample_transfer_function(self.tfn, sampleValue, sampleColor, sampleAlpha);
      opacity_correction(self, t.y - t.x, sampleAlpha);
      // blending
      alpha += (1.f - alpha) * sampleAlpha;
      return alpha < nearly_one;
    });
  }
  return 1.f - alpha;
}

template<typename T>
inline __device__ vec4f
ray_marching_traceray(const DeviceVolume<T>& self, 
                      const LaunchParams& params,
                      Ray ray) // object space ray
{
  const auto& marchingStep = self.step;

  MAYBE_UNUSED const auto& otw = *ray.otw;
  MAYBE_UNUSED const auto& wto = *ray.wto;
  auto& rng = *ray.rng;

  vec3f gradient(0);
  float alpha(0);
  vec3f color(0);

  if (intersect_box(ray.tnear, ray.tfar, ray.org, ray.dir, self.bbox)) {
    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;

    // convert ray to world coords
    const vec3f s_org = xfmPoint (otw, vec3f(ray.org));
    const vec3f s_dir = xfmVector(otw, vec3f(ray.dir));

    // start marching
    ray_marching_iterator(self, s_org, s_dir, ray.tnear, ray.tfar, marchingStep, [&](const vec2f& t) {
      assert(t.x < t.y);

      // sample data value
      const float tt = lerp(jitter, t.x, t.y);
      const auto p = ray.org + tt * ray.dir; // object space position

      // compute level of details
      const float dist = max(0.f, length(tt * ray.dir)); // distance in object space
      uint32_t lod = min(uint32_t(dist * 1.f), self.max_lod);

#ifdef ENABLE_DATA_CACHING
      float sampleValue = 0.f; 
      vec3f No;
      if (!sample_volume_gradient(self, p, &sampleValue, &No, lod)) {
        return true;
      }
#else
      // old way of sampling
      const auto sampleValue = sample_volume(self.volume, p);
      const vec3f No = -sample_gradient(self.volume, p, sampleValue, self.grad_step);
#endif

      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sample_transfer_function(self.tfn, sampleValue, sampleColor, sampleAlpha);
      opacity_correction(self, t.y - t.x, sampleAlpha);

      // access gradient
      const vec3f Nw = xfmNormal(otw, No);

      // shading
      const auto rdir = xfmVector(otw, ray.dir);

#ifndef ENABLE_DATA_CACHING
      vec3f totalShadingColor = 0.f;
      {
        Ray shadow_ray = ray;
        shadow_ray.org = p;
        shadow_ray.dir = xfmVector(wto, normalize(params.l_distant.direction));
        shadow_ray.tnear = 0.f;
        shadow_ray.tfar = float_large;
        shadow_ray.shadow = true;
        float transmittance = 0.f;
        transmittance = ray_marching_transmittance(self, params, shadow_ray, lod);
        vec3f shadingColor = 0.f;
        // TODO shade_scivis_light function seems to give wrong rendering result
        shadingColor = shade_scivis_light(rdir, Nw, sampleColor, 
            params.mat_scivis, params.l_ambient.color, 
            params.l_distant.color, params.l_distant.direction
          );
        // shadingColor = sampleColor; // fallback to not using gradient shading
        totalShadingColor += shadingColor * transmittance;
      }
      sampleColor = lerp(ray_marching_shading_scale, sampleColor, totalShadingColor);
#endif

      // compositing
      const float tr = 1.f - alpha;
      color += tr * sampleColor * sampleAlpha;
      alpha += tr * sampleAlpha;

      return alpha < nearly_one;
    });
  }

  return vec4f(color, alpha);
}

template<typename T>
__global__ void
ray_marching_kernel(uint32_t width, uint32_t height, const LaunchParams params, DeviceVolume<T> self)
{
  // compute pixel ID
  const size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width) return;
  if (iy >= height) return;

  assert(width  == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  const uint32_t pidx = ix + iy * width; // pixel index

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = self.transform;
  const affine3f wto = otw.inverse();

  // generate ray & payload
  RandomTEA rng(params.frame_index, pidx);

  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;

  ray.rng = &rng;
  ray.wto = &wto;
  ray.otw = &otw;

  // trace ray
  const vec4f output = ray_marching_traceray(self, params, ray);

  // and write to frame buffer ...
  write_pixel(params, output, pidx);
}

// ------------------------------------------------------------------
// dda iter helper
// ------------------------------------------------------------------

template<typename T>
inline __device__ bool
delta_tracking(const DeviceVolume<T>& self, Ray& ray, float& _t, vec3f& _albedo)
{
  MAYBE_UNUSED const float sigma_t = 1.f;
  MAYBE_UNUSED const float sigma_s = 1.f;
  const float majorant = self.density_scale;

  float t = ray.tnear;
  vec3f albedo(0);
  bool found_hit = false;

  while (true) {
    const float2 xi = ray.rng->get_floats();
    t = t + -logf(1.f - xi.x) / (majorant * sigma_t);
    if (t > ray.tfar) {
      found_hit = false;
      break;
    }

    const auto c = ray.org + t * ray.dir; /* object space position */
#ifdef ENABLE_DATA_CACHING
    float sample = 0.f; 
    if (!sample_volume(self, c, &sample, /*lod=*/0.0f)) { return false; }
#else
    const auto sample = sample_volume(self.volume, c);
#endif
    const auto rgba = sample_transfer_function(self.tfn.colors, self.tfn.alphas, //
                                               self.tfn.value_range.lo,          //
                                               self.tfn.value_range.hi,          //
                                               self.tfn.range_rcp_norm,          //
                                               sample);

    if (xi.y < rgba.w * self.density_scale / majorant) {
      albedo = rgba.xyz();
      found_hit = true;
      break;
    }
  }

  _t = t;
  _albedo = albedo;
  return found_hit;
}

inline __device__ bool russian_roulette(vec3f& throughput, RandomTEA& rng, const int32_t& scatter_index)
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

template<typename T>
inline __device__ vec3f
path_tracing_traceray(const DeviceVolume<T>& self, 
                      const LaunchParams& params,
                      Ray ray) // object space ray
{
  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  float t;
  vec3f albedo;

  int scatter_index = 0;
  while (intersect_box(ray.tnear, ray.tfar, ray.org, ray.dir, self.bbox)) {
    const bool exited = !delta_tracking(self, ray, t, albedo);

    if (ray.shadow) {

      if (exited) {
        L += throughput * params.l_distant.color;
      }

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(*ray.wto, uniform_sample_sphere(1.f, ray.rng->get_floats()));
      ray.shadow = false;
    }
    else {

      if (exited) {              // ray exits the volume, compute lighting
        if (scatter_index > 0) { // no light accumulation for primary rays
          L += throughput * params.l_ambient.color;
        }
        break;
      }

      if (russian_roulette(throughput, *ray.rng, scatter_index)) break;
      ++scatter_index;

      ray.org = ray.org + t * ray.dir;
      throughput *= /* phase function */(albedo);

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(*ray.wto, normalize(params.l_distant.direction));
      ray.shadow = true;
    }
  }

  return L;
}

template<typename T>
__global__ void
path_tracing_kernel(uint32_t width, uint32_t height, const LaunchParams params, DeviceVolume<T> self)
{
  // compute pixel ID
  const size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width) return;
  if (iy >= height) return;

  assert(width  == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  const uint32_t pidx = ix + iy * width; // pixel index

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = self.transform;
  const affine3f wto = otw.inverse();

  // generate ray & payload
  RandomTEA rng(params.frame_index, pidx);

  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;

  ray.rng = &rng;
  ray.wto = &wto;
  ray.otw = &otw;

  // trace ray
  const vec3f color = path_tracing_traceray(self, params, ray);

  // and write to frame buffer ... (accumilative)
  write_pixel(params, vec4f(color, 1.f), pidx);
}

template<typename T>
static void 
render_templated(cudaStream_t stream, LaunchParams& params, StructuredRegularVolume::Device& grid, tdns::gpucache::CacheManager<T> *manager, uint32_t max_lod) 
{
  const auto& size = params.frame.size;

  const int n_threads_bilinear = 16;
	const dim3 block_size(n_threads_bilinear, n_threads_bilinear, 1);
  const dim3 grid_size(
    misc::div_round_up(size.x, n_threads_bilinear), 
    misc::div_round_up(size.y, n_threads_bilinear), 
    1
  );

  DeviceVolume<T> volume(grid, manager->to_kernel_object());
  volume.max_lod = max_lod;

  // this renders a single frame
  if (params.enable_path_tracing) {
  	path_tracing_kernel<<<grid_size, block_size, 0, stream>>>(size.x, size.y, params, volume);
  } else {
    ray_marching_kernel<<<grid_size, block_size, 0, stream>>>(size.x, size.y, params, volume);
  }

  // manager->update ... 
  manager->update();
  std::vector<float> completude;
  manager->completude(completude);
  std::cout << "3DNS - [Cache used " << std::setw(6) << std::to_string(completude[0] * 100.f) << "%]\r";
}

void 
MethodMegakernel::render(cudaStream_t stream, LaunchParams& params, StructuredRegularVolume::Device& grid, OpaqueCacheManager& manager, bool reset_accumulation)
{
  if (params.frame.size.x <= 0 || params.frame.size.y <= 0) return;

  // resize accumulation buffer if necessary (if size doesnot change, it will not reallocate, so it is safe to call it every frame)
  accumulation_buffer.resize(params.frame.size.long_product() * sizeof(vec4f) /*, stream*/);
  if (reset_accumulation) {
    accumulation_buffer.nullify(stream);
  }
  params.accumulation = (vec4f*)accumulation_buffer.d_pointer();

  // upload all the params to the GPU
  params_buffer.resize(sizeof(params));
  params_buffer.upload_async(&params, 1, stream);

  // switch (manager.type) {
  // case VALUE_TYPE_UINT8:  return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<uchar1> *)manager.cache, manager.max_lod);
  // case VALUE_TYPE_INT8:   return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<char1>  *)manager.cache, manager.max_lod);
  // case VALUE_TYPE_UINT16: return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<ushort1>*)manager.cache, manager.max_lod);
  // case VALUE_TYPE_INT16:  return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<short1> *)manager.cache, manager.max_lod);
  // case VALUE_TYPE_UINT32: return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<uint1>  *)manager.cache, manager.max_lod);
  // case VALUE_TYPE_INT32:  return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<int1>   *)manager.cache, manager.max_lod);
  // case VALUE_TYPE_FLOAT:  return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<float1> *)manager.cache, manager.max_lod);
  // default: throw std::runtime_error("unsupported type encountered: " + std::to_string(manager.type));
  // }
  return render_templated(stream, params, grid, (tdns::gpucache::CacheManager<float1> *)manager.cache, manager.max_lod);
}

}
} // namespace ovr
