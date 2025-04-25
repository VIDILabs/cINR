#pragma once
#ifndef OVR_NNCACHE_DEVICE_H
#define OVR_NNCACHE_DEVICE_H

#include "ovr/renderer.h"

#include "array.h"
#include "framebuffer.h"

#include "serializer.h"
#include "renderer/renderer.h"

#include <memory>

namespace ovr::nncache {

using FrameBuffer  = DoubleBufferObject<vec4f>;

struct OpaqueCacheManager;

class DeviceNNCache : public MainRenderer {
public:
  /*! constructor - performs all setup, including initializing ospray, creates scene graph, etc. */
  void init(int argc, const char** argv) override;

  /*! render one frame */
  void swap() override;
  void commit() override;
  void render() override;
  void mapframe(FrameBufferData* fb) override;

  /*! control device specific UIs */
  void ui() override;

private:
  void commit_material();
  void commit_lighting();

  template<typename T>
  bool check(TransactionalValue<T>& ctl) {
    if (ctl.update()) { dirty = true; return true; }
    return false;
  }

private:
  bool initialized = false;
  bool dirty = true;

  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  bool framebuffer_size_updated{ false };

  Camera camera;

  StructuredRegularVolume volume;
  bool volume_changed{ true };

  std::shared_ptr<OpaqueCacheManager> cacheManager;

  bool waverfront = true;
  int rendering_mode{ 0 };
  RenderObject api;
  LaunchParams &lp = api.params;

  struct {
    TransactionalValue<bool> wavefront;
    TransactionalValue<int> cachemode;

    TransactionalValue<float> lod_scale;
    TransactionalValue<float> lod_threshold;

    TransactionalValue<float> ambient;
    TransactionalValue<float> diffuse;
    TransactionalValue<float> specular;
    TransactionalValue<float> shininess;
    TransactionalValue<float> phi;
    TransactionalValue<float> theta;
    TransactionalValue<float> intensity;
  } ctls;
};

} // namespace ovr::nncache

#endif // OVR_NNCACHE_DEVICE_H
