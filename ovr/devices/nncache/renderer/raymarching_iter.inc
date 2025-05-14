#pragma once

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

inline __device__ float
sample_size_scaler(const float ss, const float t0, const float t1) {
  const int32_t N = (t1-t0) / ss + 1;
  return (t1-t0) / N;
}

inline __device__
RayMarchingIter<true>::RayMarchingIter(const VNRDeviceVolume& self, const Ray& ray)
{
  const auto org = ray.org;
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  DDAIter::init(m_org, m_dir, tMin, tMax, dims);
}

template<typename F>
inline __device__ void
RayMarchingIter<true>::exec(const VNRDeviceVolume& self, const Ray& ray, const float step, const uint32_t pidx, const F& body)
{
  const auto org = ray.org;
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;

  const auto lambda = [&](const vec3i& cell, float t0, float t1) {
    // calculate max opacity
    float r = opacityUpperBound(self, cell);
    // float r = 0.1;
    // if (r < 0.027032 || r > 0.027034)
    //   printf("cell: (%i, %i, %i) | r: %f \n\n", cell.x, cell.y, cell.z, r);
    if (fabsf(r) <= float_epsilon) 
    {
      // printf("Cell is empty\n\n");
      return true; // the cell is empty 
    }
    // estimate a step size
    // auto ss = adaptiveSamplingRate(step, r);
    float ss = sample_size_scaler(adaptiveSamplingRate(step, r), t0, t1);
    // printf("ss: %f\n\n", ss);
    // ss += ray.jitter <= 0.5 ? (ss*ray.jitter) : 0;
    // t0 += ray.jitter <= 0.01 ? 0 : ray.jitter >= 0.99 ? ss : ss * ray.jitter;
    
    // iterate within the interval
    vec2f t = vec2f(t0, min(t1, t0 + ss));
    while (t.y > t.x) {
      DDAIter::next_cell_begin = t.y - tMin;
      if (!body(t, r)) return false;
      t.x = t.y;
      t.y = min(t.x + ss, t1);
    }
    return true;
  };

  while (DDAIter::next(m_org, m_dir, tMin, tMax, dims, false, lambda)) {}
}

inline __device__ bool 
RayMarchingIter<true>::resumable(const VNRDeviceVolume& self, const Ray& ray)
{
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  const auto& dims = self.macrocell_dims;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  return DDAIter::resumable(m_dir, tMin, tMax, dims);
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------


inline __device__
RayMarchingIter<false>::RayMarchingIter(const VNRDeviceVolume& self, const Ray& ray)
{
}

template<typename F>
inline __device__ void
RayMarchingIter<false>::exec(const VNRDeviceVolume& self, const Ray& ray, const float step, const uint32_t pidx, const F& body)
{
  const auto org = ray.org;
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  vec2f t;
  t.x = max(tMin + next_cell_begin, tMin);
  t.y = min(t.x + step, tMax);
  while (t.y > t.x) {
    next_cell_begin = t.y - tMin;
    if (!body(t)) return;
    t.x = t.y;
    t.y = min(t.x + step, tMax);
  }
  next_cell_begin = float_large;
  return;
}

inline __device__ bool 
RayMarchingIter<false>::resumable(const VNRDeviceVolume& self, const Ray& ray)
{
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  return tMin + next_cell_begin < tMax;
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

// inline __device__ float
// sample_size_scaler(const float ss, const float t0, const float t1) {
//   const int32_t N = (t1-t0) / ss + 1;
//   return (t1-t0) / N;
// }

template<typename F>
inline __device__ void
raymarching_iterator_0(const VNRDeviceVolume& self, 
                     const vec3f& org, const vec3f& dir,
                     const float tMin, const float tMax, 
                     const float step, 
                     const F& body, 
                     bool debug)
{
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  dda::dda3(m_org, m_dir, tMin, tMax, dims, debug, [&] __device__ (const vec3i& cell, float t0, float t1) {
    // calculate max opacity
    float r = opacityUpperBound(self, cell);
    if (fabsf(r) <= float_epsilon) return true; // the cell is empty
    // estimate a step size
    const auto ss = sample_size_scaler(adaptiveSamplingRate(step, r), t0, t1);
    // iterate within the interval
    vec2f t = vec2f(t0, min(t1, t0 + ss));
    while (t.y > t.x) {
      if (!body(t)) return false;
      t.x = t.y;
      t.y = min(t.x + ss, t1);
    }
    return true;
  });
}

template<typename F>
inline __device__ void
raymarching_iterator_1(const VNRDeviceVolume& self, 
                     const vec3f& org, const vec3f& dir,
                     const float tMin, const float tMax, 
                     const float step, 
                     const F& body, 
                     bool debug)
{
  vec2f t = vec2f(tMin, min(tMax, tMin + step));
  while ((t.y > t.x) && body(t)) {
    t.x = t.y;
    t.y = min(t.x + step, tMax);
  }
}
