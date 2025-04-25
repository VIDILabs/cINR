#pragma once

#include "../instantvnr_types.h"

#include <memory>

namespace vnr {

struct SamplerAPI // SamplerImpl
{
private:
  vec3i m_rendering_dims{};
  affine3f m_transform;

public:
  typedef ValueType dtype;

  virtual ~SamplerAPI() = default;
  virtual void set_current_volume_timestamp(int index) { if (index != 0) throw std::runtime_error("only support single timestep volume"); }
  virtual cudaTextureObject_t texture() const = 0;
  virtual dtype type() const = 0;
  virtual vec3i dims() const = 0;
  // virtual float lower() const = 0;
  // virtual float upper() const = 0;
  virtual void sample(void* d_coords, void* d_values, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream) = 0;
  virtual void sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) {};

  vec3i rendering_dims() const { return m_rendering_dims; }
  void set_rendering_dims(const vec3i& dims) { m_rendering_dims = dims; }

  affine3f transform() const { return m_transform; }
  void set_transform(const affine3f& xfm) { m_transform = xfm; }

  void set_current_volume_index(int index) { set_current_volume_timestamp(index); }

  void take_samples(void* d_input, void* d_output, size_t num_samples, cudaStream_t stream, const vec3f& lower, const vec3f& upper) {
    sample(d_input, d_output, num_samples, lower, upper, stream);
  }
  void take_samples_grid(void* d_input, void* d_output, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) {
    sample_grid(d_input, d_output, grid_origin, grid_dims, grid_spacing, stream);
  }

  static std::shared_ptr<SamplerAPI> 
  create(const VolumeDesc& desc, std::string training_mode, bool save_volume = false);
};

typedef std::shared_ptr<SamplerAPI> Sampler;

}
