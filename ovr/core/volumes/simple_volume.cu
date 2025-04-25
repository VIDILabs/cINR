#include "volumes.h"

#include "samplers/neural_sampler.h"

INSTANT_VNR_NAMESPACE_BEGIN

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
SimpleVolume::load(const VolumeDesc& descriptor, std::string sampling_mode, bool save_volume)
{
  desc = descriptor;
  mode = sampling_mode;
  sampler = SamplerAPI::create(desc, sampling_mode, save_volume);
  tex = sampler->texture();
  if (tex) {
    macrocell.set_shape(desc.dims);
    macrocell.allocate();
    macrocell.compute_everything(tex);
  }
}

void 
SimpleVolume::load(const void* data, vec3i dims, std::string type, range1f range, std::string sampling_mode)
{
  mode = sampling_mode;

  desc.dims = dims;
  desc.type = value_type(type);
  desc.range = range;

  sampler = std::make_shared<CudaSampler>(data, desc.dims, desc.type, desc.range, true);
  sampler->set_rendering_dims(sampler->dims());
  sampler->set_transform(affine3f::translate(vec3f(desc.dims) * -0.5f) * affine3f::scale(vec3f(desc.dims)));

  tex = sampler->texture();
  if (tex) {
    macrocell.set_shape(desc.dims);
    macrocell.allocate();
    macrocell.compute_everything(tex);
  }
}

void 
SimpleVolume::set_current_timestep(int index) 
{ 
  sampler->set_current_volume_index(index); 
  if (tex && !macrocell.is_external()) {
    macrocell.compute_everything(tex);
  }
}

void SimpleVolume::set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r)
{
  tfn.set_transfer_function(c, o, r, nullptr);
  if (macrocell.allocated()) {
    macrocell.update_max_opacity(tfn.tfn, nullptr);
  }
}

void SimpleVolume::set_data_transform(affine3f transform)
{
  sampler->set_transform(transform);
}

INSTANT_VNR_NAMESPACE_END
