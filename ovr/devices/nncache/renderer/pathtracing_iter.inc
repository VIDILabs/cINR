#pragma once

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

inline __device__ 
PathTracingIter<true>::PathTracingIter() {}

inline __device__ 
PathTracingIter<true>::PathTracingIter(const VNRDeviceVolume& self, const Ray& ray) 
{
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  DDAIter::init(m_org, m_dir, ray.tnear, ray.tfar, dims);
}

inline __device__ bool 
PathTracingIter<true>::hashit(const VNRDeviceVolume& self, const Ray& ray, float& rayt, float& majorant) 
{
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;

  const auto& density_scale = self.density_scale;
  MAYBE_UNUSED const float sigma_t = 1.f;
  MAYBE_UNUSED const float sigma_s = 1.f;

  bool found_hit = false;
  float tau = -logf(1.f - ray.rng.get_float());
  float t = next_cell_begin + ray.tnear;

  while (DDAIter::next(m_org, m_dir, ray.tnear, ray.tfar, dims, /*dbg=*/false, 
        [&](const vec3i& c, float t0, float t1) 
        {
          majorant = opacityUpperBound(self, c) * density_scale;
          if (fabsf(majorant) <= float_epsilon) return true; // move to the next macrocell
          tau -= (t1 - t) * (majorant * sigma_t); t = t1;
          if (tau > 0.f) return true; // move to the next macrocell  
          t = t + tau / (majorant * sigma_t); // can have division by zero error
          found_hit = true;
          next_cell_begin = t - ray.tnear;
          rayt = t;
          return false; // found a hit, terminate the loop
        }
      )
    ) 
  {}

  return found_hit;
}

inline __device__ bool 
PathTracingIter<true>::finished(const VNRDeviceVolume& self, const Ray& ray) {
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  return !DDAIter::resumable(m_dir, ray.tnear, ray.tfar, dims);   
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

inline __device__ 
PathTracingIter<false>::PathTracingIter() {}

inline __device__ 
PathTracingIter<false>::PathTracingIter(const VNRDeviceVolume& self, const Ray& ray) : t(0) {}

inline __device__ bool 
PathTracingIter<false>::hashit(const VNRDeviceVolume& self, const Ray& ray, float& rayt, float& majorant) 
{
  MAYBE_UNUSED const float sigma_t = 1.f;
  MAYBE_UNUSED const float sigma_s = 1.f;
  majorant = self.density_scale;
  t += -logf(1.f - ray.rng.get_float()) / (majorant * sigma_t);
  rayt = ray.tnear + t;
  return (rayt <= ray.tfar);
}

inline __device__ bool 
PathTracingIter<false>::finished(const VNRDeviceVolume& self, const Ray& ray) {
  return (ray.tnear + t > ray.tfar);
}


// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

template<>
inline __device__ bool
delta_tracking<true>(const VNRDeviceVolume& self, const Ray& ray, /* object space ray */ float& _t, vec3f& _albedo)
{
  const auto density_scale = self.density_scale;
  MAYBE_UNUSED const float sigma_t = 1.f;
  MAYBE_UNUSED const float sigma_s = 1.f;

  float t = ray.tnear;
  vec3f albedo(0);
  bool found_hit = false;

  float majorant;
  Iter iter(self, ray);
  while (iter.hashit(self, ray, t, majorant)) {
    const auto c = ray.org + t * ray.dir; // object space position
    const auto sample = sampleVolume(self.volume, c);
    const auto rgba = sampleTransferFunction(self.tfn, sample);
    if (ray.rng.get_float() * majorant < rgba.w * density_scale) {
      albedo = rgba.xyz();
      found_hit = true;
      break;
    }
  }

  _t = t;
  _albedo = albedo;
  return found_hit;
}

template<>
inline __device__ bool
delta_tracking<false>(const VNRDeviceVolume& self, const Ray& ray, /* object space ray */ float& _t, vec3f& _albedo)
{
  const auto density_scale = self.density_scale;
  MAYBE_UNUSED const float sigma_t = 1.f;
  MAYBE_UNUSED const float sigma_s = 1.f;

  float t = ray.tnear;
  vec3f albedo(0);
  bool found_hit = false;

  const auto majorant = density_scale;
  while (true) {
    const vec2f xi = ray.rng.get_floats();
    t = t + -logf(1.f - xi.x) / (majorant * sigma_t);
    if (t > ray.tfar) {
      found_hit = false;
      break;
    }
    const auto c = ray.org + t * ray.dir; // object space position
    const auto sample = sampleVolume(self.volume, c);
    const auto rgba = sampleTransferFunction(self.tfn, sample);
    if (xi.y < rgba.w * density_scale / majorant) {
      albedo = rgba.xyz();
      found_hit = true;
      break;
    }
  }

  _t = t;
  _albedo = albedo;
  return found_hit;
}
