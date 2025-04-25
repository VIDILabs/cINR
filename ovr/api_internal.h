
#pragma once

#include "api.h"

#include "core/instantvnr_types.h"
#include "core/volumes/volumes.h"
#include "serializer.h"

namespace vnr {

using namespace vnr::math;

struct VolumeContext 
{
  vec3i     dims;
  ValueType type;
  range1f   range;
  box3f clipbox;
  virtual ~VolumeContext() {};
  virtual bool isNetwork() const = 0;
};

struct SimpleVolumeContext : VolumeContext 
{
  SimpleVolume source;

  SimpleVolumeContext() = default;

  bool isNetwork() const override { return false; };
};

struct NeuralVolumeContext : VolumeContext 
{
  NeuralVolume neural;

  NeuralVolumeContext(size_t batchsize) : neural(batchsize) {}

  bool isNetwork() const override { return true; };
};

}
