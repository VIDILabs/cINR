#pragma once

#include "core/instantvnr_types.h"

#include <json/json.hpp>

#include <fstream>
#include <string>

namespace vnr {

using json = nlohmann::json;

struct CacheConfig {
  std::string config;
  vec3i capacity = vec3i(16);
  int num_levels = 1;

  std::string macrocell;
  vec3i macrocell_dims;
  vec3f macrocell_spacings;
};

void
create_json_scene_stringify(const json& root, TransferFunction& tfn, Camera& camera, vec3i& dims, ValueType& type);

void
create_json_scene_stringify(const json& root, VolumeDesc& volume, TransferFunction& tfn, Camera& camera);

inline void
create_json_scene(std::string filename, VolumeDesc& volume, TransferFunction& tfn, Camera& camera)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_scene_stringify(root, volume, tfn, camera);
}

void
create_json_tfn_stringify(const json& root, TransferFunction& tfn);

inline void
create_json_tfn(std::string filename, TransferFunction& tfn)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_tfn_stringify(root, tfn);
}

void
create_json_volume_stringify(const json& root, VolumeDesc& volume);

inline void
create_json_volume(std::string filename, VolumeDesc& volume)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_volume_stringify(root, volume);
}

void
create_json_camera_stringify(const json& root, Camera& camera);

inline void
create_json_camera(std::string filename, Camera& camera)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_camera_stringify(root, camera);
}

} // namespace vnr
