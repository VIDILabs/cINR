#include "sampler.h"
#include "samplers/neural_sampler.h"

INSTANT_VNR_NAMESPACE_BEGIN

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

Sampler
SamplerAPI::create(const VolumeDesc& desc, std::string training_mode, bool save_volume)
{
  Sampler impl;

  /* GPU-based */
  if (training_mode == "GPU") {
    impl = std::make_shared<CudaSampler_TimeVarying>(desc, save_volume, false);
    impl->m_rendering_dims = impl->dims();
  }

#ifdef ENABLE_OUT_OF_CORE

  /* CPU-based, virtual memory, no ground truth */
  else if (training_mode == "VIRTUAL_MEMORY") {
    impl = std::make_shared<VirtualMemorySampler>(desc);
    impl->m_rendering_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

  /* out-of-core-steaming */
  else if (training_mode == "OUT_OF_CORE") {
    impl = std::make_shared<OutOfCoreSampler>(desc);
    impl->m_rendering_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

#endif // ENABLE_OUT_OF_CORE

  else if (training_mode == "NOTHING") {
    impl = std::make_shared<DummySampler>(desc.dims);
    impl->m_rendering_dims = impl->dims();
  }

  else throw std::runtime_error("unknown mode: " + training_mode);

  impl->m_transform = affine3f::translate(vec3f(desc.dims) * -0.5f) * affine3f::scale(vec3f(desc.dims));
  return impl;
}

INSTANT_VNR_NAMESPACE_END
