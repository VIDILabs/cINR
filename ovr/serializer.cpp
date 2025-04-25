#include "serializer.h"

namespace tfn {
typedef vnr::math::vec2f vec2f;
typedef vnr::math::vec2i vec2i;
typedef vnr::math::vec3f vec3f;
typedef vnr::math::vec3i vec3i;
typedef vnr::math::vec4f vec4f;
typedef vnr::math::vec4i vec4i;
} // namespace tfn
#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
#include "tfn/core.h"

// JSON I/O
#include <json/json.hpp>

#include <colormap.h>

namespace vnr { // clang-format off

struct SingleVolumeDesc : VolumeDesc::File 
{
  vec3i dims;
  ValueType type;
};

NLOHMANN_JSON_SERIALIZE_ENUM(ValueType, {
  { ValueType::VALUE_TYPE_INT8, "BYTE" },
  { ValueType::VALUE_TYPE_UINT8, "UNSIGNED_BYTE" },
  { ValueType::VALUE_TYPE_INT16, "SHORT" },
  { ValueType::VALUE_TYPE_UINT16, "UNSIGNED_SHORT" },
  { ValueType::VALUE_TYPE_INT32, "INT" },
  { ValueType::VALUE_TYPE_UINT32, "UNSIGNED_INT" },
  { ValueType::VALUE_TYPE_FLOAT, "FLOAT" },
  { ValueType::VALUE_TYPE_DOUBLE, "DOUBLE" },
}); // clang-format on

enum Endianness { VNR_LITTLE_ENDIAN, VNR_BIG_ENDIAN };
NLOHMANN_JSON_SERIALIZE_ENUM(Endianness, {
  { VNR_LITTLE_ENDIAN, "LITTLE_ENDIAN" },
  { VNR_BIG_ENDIAN, "BIG_ENDIAN" },
}); // clang-format on

#define define_vector_serialization(T)                      \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec2##T, x, y);       \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec3##T, x, y, z);    \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec4##T, x, y, z, w); \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(range1##T, minimum, maximum);
define_vector_serialization(i);
define_vector_serialization(f);
#undef define_vector_serialization

#define assert_throw(x, msg) { if (!(x)) throw std::runtime_error(msg); }

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in);

#define define_scalar_serialization(T) template<> inline T scalar_from_json<T>(const json& in) { return in.get<T>(); }
define_scalar_serialization(std::string);
define_scalar_serialization(bool);
define_scalar_serialization(int64_t);
define_scalar_serialization(uint64_t);
define_scalar_serialization(double);

template<typename ScalarT/*, typename std::enable_if_t<!std::is_arithmetic<ScalarT>::value> = true*/>
inline ScalarT
scalar_from_json(const json& in)
{
  ScalarT v;
  from_json(in, v);
  return v;
}

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in, const std::string& key)
{
  assert_throw(in.is_object(), "has to be a JSON object");
  assert_throw(in.contains(key), "incorrect key: " + key);
  return scalar_from_json<ScalarT>(in[key]);
}

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in, const std::string& key, const ScalarT& value)
{
  assert_throw(in.is_object(), "has to be a JSON object");
  if (in.contains(key)) {
    return scalar_from_json<ScalarT>(in[key]);
  }
  else {
    return value;
  }
}

namespace {

static vec2f
rangeFromJson(const json& jsrange)
{
  if (!jsrange.contains("minimum") || !jsrange.contains("maximum")) {
    return vec2f(0.0, 0.0);
  }
  return vec2f(jsrange["minimum"].get<float>(), jsrange["maximum"].get<float>());
}

static bool
file_exists_test(const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

static std::string
valid_filename(const json& in, const std::string& key)
{
  if (in.contains(key)) {
    auto& js = in[key];
    if (js.is_array()) {
      for (auto& s : js) {
        if (file_exists_test(s.get<std::string>())) return s.get<std::string>();
      }
      // throw std::runtime_error("Cannot find volume file.");
      std::cerr << "Cannot find volume file." << std::endl;
      return "INVALID";
    }
    else {
      return js.get<std::string>();
    }
  }
  else {
    throw std::runtime_error("Json key 'fileName' doesnot exist");
  }
}

} // namespace

void create_json_volume_stringify_diva(const json& root, VolumeDesc& volume)
{
  const auto config = root["volume"];
  const vec2f range = scalar_from_json<vec2f>(config, "range");

  volume.dims = scalar_from_json<vec3i>(config["dims"]);
  volume.type = scalar_from_json<ValueType>(config["type"]);
  volume.range.lower = range.x;
  volume.range.upper = range.y;

  if (config["filename"].is_array()) {
    for (auto& v : config["filename"]) {
      VolumeDesc::File f;
      f.filename = v.get<std::string>();
      f.bigendian = config.contains("bigendian") ? config["bigendian"].get<bool>() : false;
      f.offset = 0;
      volume.data.push_back(f);
    }
  }

  else {
    VolumeDesc::File file;
    file.filename = config["filename"].get<std::string>();
    file.bigendian = config.contains("bigendian") ? config["bigendian"].get<bool>() : false;
    file.offset = 0;
    volume.data.push_back(file);
  }
  
  volume.scale = vec3f(volume.dims);
  volume.translate = -vec3f(volume.dims) / 2.f;
}

void
create_json_scene_diva(const json& root, VolumeDesc& volume, TransferFunction& tfn, Camera& camera)
{
  create_json_volume_stringify_diva(root, volume);
  // TODO load TFN and Camera //
  throw std::runtime_error("not implemented");
}

Camera
create_scene_vidi__camera(const json& jscamera)
{
  Camera camera;
  camera.from = scalar_from_json<vec3f>(jscamera["eye"]);
  camera.at = scalar_from_json<vec3f>(jscamera["center"]);
  camera.up = scalar_from_json<vec3f>(jscamera["up"]);
  camera.fovy = jscamera["fovy"].get<float>();
  return camera;
}

TransferFunction
create_scene_vidi__tfn(const json& jstfn, const json& jsvolume, ValueType type)
{
  TransferFunction tfn;

  /* load transfer function */
  if (jstfn.is_null()) {
    const auto& ctable = (std::vector<vec4f>&)colormap::get("diverging/RdBu");
    std::vector<vec3f> color(ctable.size());
    for (int i = 0; i < ctable.size(); ++i) {
      color[i] = ctable[i].xyz();
    }
    tfn.color = std::move(color);
    tfn.alpha = { vec2f(0.f, 0.f), vec2f(1.f, 1.f) };
  }
  else {
    tfn::TransferFunctionCore tf;
    tfn::loadTransferFunction(jstfn, tf);
    // convert to our format
    auto* table = (vec4f*)tf.data();
    std::vector<vec3f> color(tf.resolution());
    std::vector<vec2f> alpha(tf.resolution());
    for (int i = 0; i < tf.resolution(); ++i) {
      auto rgba = table[i];
      color[i] = rgba.xyz();
      alpha[i] = vec2f((float)i / (tf.resolution() - 1), rgba.w);
    }
    if (alpha[0].y < 0.01f) alpha[0].y = 0.f;
    if (alpha[tf.resolution()-1].y < 0.01f) alpha[tf.resolution()-1].y = 0.f;
    tfn.color = std::move(color);
    tfn.alpha = std::move(alpha);
  }

  /* transfer function value range */
  if (jsvolume.contains("scalarMappingRangeUnnormalized")) {
    auto r = rangeFromJson(jsvolume["scalarMappingRangeUnnormalized"]);
    tfn.range.lower = r.x;
    tfn.range.upper = r.y;
  }
  else if (jsvolume.contains("scalarMappingRange")) {
    auto r = rangeFromJson(jsvolume["scalarMappingRange"]);
    switch (type) {
    case VALUE_TYPE_UINT8:
      tfn.range.lower = r.x * (std::numeric_limits<uint8_t>::max() - std::numeric_limits<uint8_t>::lowest()) + std::numeric_limits<uint8_t>::lowest();
      tfn.range.upper = r.y * (std::numeric_limits<uint8_t>::max() - std::numeric_limits<uint8_t>::lowest()) + std::numeric_limits<uint8_t>::lowest();
      break;
    case VALUE_TYPE_INT8:
      tfn.range.lower = r.x * (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::lowest()) + std::numeric_limits<int8_t>::lowest();
      tfn.range.upper = r.y * (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::lowest()) + std::numeric_limits<int8_t>::lowest();
      break;
    case VALUE_TYPE_UINT16:
      tfn.range.lower = r.x * (std::numeric_limits<uint16_t>::max() - std::numeric_limits<uint16_t>::lowest()) + std::numeric_limits<uint16_t>::lowest();
      tfn.range.upper = r.y * (std::numeric_limits<uint16_t>::max() - std::numeric_limits<uint16_t>::lowest()) + std::numeric_limits<uint16_t>::lowest();
      break;
    case VALUE_TYPE_INT16:
      tfn.range.lower = r.x * (std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::lowest()) + std::numeric_limits<int16_t>::lowest();
      tfn.range.upper = r.y * (std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::lowest()) + std::numeric_limits<int16_t>::lowest();;
      break;
    // NOTE: in NN Cache project, we assume CUDA-style data normalization rule:
    // -- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
    //    about cudaTextureDesc::readMode:
    //       Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted,
    //       regardless of whether or not this cudaTextureDesc::readMode is set cudaReadModeNormalizedFloat is specified.
    case VALUE_TYPE_UINT32:
    case VALUE_TYPE_INT32:
    case VALUE_TYPE_FLOAT:
    case VALUE_TYPE_DOUBLE:
      tfn.range.lower = r.x;
      tfn.range.upper = r.y;
      break;
    default: throw std::runtime_error("unknown data type");
    }
  }
  else {
    switch (type) {
    case VALUE_TYPE_UINT8:
      tfn.range.lower = std::numeric_limits<uint8_t>::max();
      tfn.range.upper = std::numeric_limits<uint8_t>::lowest();
      break;
    case VALUE_TYPE_INT8:
      tfn.range.lower = std::numeric_limits<int8_t>::max();
      tfn.range.upper = std::numeric_limits<int8_t>::lowest();
      break;
    case VALUE_TYPE_UINT16:
      tfn.range.lower = std::numeric_limits<uint16_t>::max();
      tfn.range.upper = std::numeric_limits<uint16_t>::lowest();
      break;
    case VALUE_TYPE_INT16:
      tfn.range.lower = std::numeric_limits<int16_t>::max();
      tfn.range.upper = std::numeric_limits<int16_t>::lowest();
      break;
    default: {
      // throw std::runtime_error("[Error] value range is not provided, this is required for 32-bit integers and floating point data");
      std::cerr << "[vidi] value range is not provided, this is required for 32-bit integers and floating point data" << std::endl;
      tfn.range.lower = std::numeric_limits<float>::max();
      tfn.range.upper = std::numeric_limits<float>::lowest();
    }
    }
  }
  return tfn;
}

SingleVolumeDesc
create_scene_vidi__volume(const json& jsdata)
{
  SingleVolumeDesc volume;

  const auto format = scalar_from_json<std::string>(jsdata["format"]);

  if (format == "REGULAR_GRID_RAW_BINARY") {
    const auto filename      = valid_filename(jsdata, "fileName");
    const auto dims          = scalar_from_json<vec3i>(jsdata, "dimensions");
    const auto type          = scalar_from_json<ValueType>(jsdata, "type");
    const auto offset        = scalar_from_json<size_t>(jsdata, "offset", 0);
    const auto flipped       = scalar_from_json<bool>(jsdata, "fileUpperLeft", false);
    const auto is_big_endian = scalar_from_json<Endianness>(jsdata, "endian", VNR_LITTLE_ENDIAN) == VNR_BIG_ENDIAN;
    volume.filename = filename;
    volume.offset = offset;
    volume.bigendian = is_big_endian;
    volume.dims = dims;
    volume.type = type;
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }

  return volume;
}

VolumeDesc::File
create_scene_vidi__multivolume(const json& jsdata, const SingleVolumeDesc& volume)
{
  VolumeDesc::File file;

  const auto format = scalar_from_json<std::string>(jsdata["format"]);

  if (format == "REGULAR_GRID_RAW_BINARY") {
    const auto filename      = valid_filename(jsdata, "fileName");
    const auto dims          = scalar_from_json<vec3i>(jsdata, "dimensions");
    const auto type          = scalar_from_json<ValueType>(jsdata, "type");
    const auto offset        = scalar_from_json<size_t>(jsdata, "offset", 0);
    const auto flipped       = scalar_from_json<bool>(jsdata, "fileUpperLeft", false);
    const auto is_big_endian = scalar_from_json<Endianness>(jsdata, "endian", VNR_LITTLE_ENDIAN) == VNR_BIG_ENDIAN;
    assert(volume.type == type);
    assert(volume.dims == dims);
    file.filename = filename;
    file.offset = offset;
    file.bigendian = is_big_endian;
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }

  return file;
}

ValueType
create_scene_vidi__datatype(const json& jsdata)
{
  const auto format = scalar_from_json<std::string>(jsdata["format"]);
  if (format == "REGULAR_GRID_RAW_BINARY") {
    return scalar_from_json<ValueType>(jsdata, "type");
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }
}

void
create_json_scene_stringify(const json& root, TransferFunction& tfn, Camera& camera, vec3i& dims, ValueType& type) 
{
  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");

  // reate primary volume
  SingleVolumeDesc pv = create_scene_vidi__volume(ds[0]);

  // create multi volume
  dims = pv.dims;
  type = pv.type;

  // load the transfer function as well as the value range
  if (root["view"]["volume"].contains("transferFunction")) {
    tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], pv.type);
  }
  else {
    tfn = create_scene_vidi__tfn(json(), root["view"]["volume"], pv.type);
  }

  // when an integer volume is being used, the data value will be normalized, if an unnormalized value range is not present, we produce a warning
  if (!root["view"]["volume"].contains("scalarMappingRangeUnnormalized")) {
    if (type != VALUE_TYPE_FLOAT && type != VALUE_TYPE_DOUBLE) {
      std::cerr << "[vidi] An unnormalized value range cannot be found for transfer function, incorrect results can be produced." << std::endl;
    }
  }

  // ovr camera setting (do not centralize camera position)
  if (root["view"].contains("camera")) {
    camera = create_scene_vidi__camera(root["view"]["camera"]);
  }
  else {
    // create a default camera
    camera.from = vec3f(dims) / 2.f;
    camera.from.z += 2.f * dims.z;
    camera.at = vec3f(dims) / 2.f;
    camera.up = vec3f(0.f, 1.f, 0.f);
  }
}

void
create_json_scene_vidi(const json& root, VolumeDesc& volume, TransferFunction& tfn, Camera& camera)
{
  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");

  // reate primary volume
  SingleVolumeDesc pv = create_scene_vidi__volume(ds[0]);

  // create multi volume
  volume.dims = pv.dims;
  volume.type = pv.type;
  volume.data.resize(ds.size()); 
  volume.data[0] = pv; // slicing
  for (int i = 1; i < ds.size(); ++i) {
    volume.data.push_back(create_scene_vidi__multivolume(ds[i], pv));
  }

  // load the transfer function as well as the value range
  tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], pv.type);
  volume.range = tfn.range;

  // when an integer volume is being used, the data value will be normalized, if an unnormalized value range is not present, we produce a warning
  if (!root["view"]["volume"].contains("scalarMappingRangeUnnormalized")) {
    auto type = scalar_from_json<ValueType>(ds[0]["type"]);
    if (type != VALUE_TYPE_FLOAT && type != VALUE_TYPE_DOUBLE) {
      std::cerr << "[vidi] An unnormalized value range cannot be found for transfer function, incorrect results can be produced." << std::endl;
    }
  }

  // create camera
  camera = create_scene_vidi__camera(root["view"]["camera"]);
  camera.at   -= vec3f(volume.dims) / 2.f;
  camera.from -= vec3f(volume.dims) / 2.f;
}

// void
// create_json_data_type_stringify_vidi(json root, ValueType& type)
// {
//   const auto& ds = root["dataSource"];
//   assert_throw(ds.is_array(), "'dataSource' is expected to be an array");
//   assert_throw(ds.size() >= 1, "'dataSource' should contain at least one element");
// 
//   type = create_scene_vidi__datatype(ds[0]);
// }

void
create_json_tfn_stringify_vidi(const json& root, TransferFunction& tfn)
{
  ValueType type; // create_json_data_type_stringify_vidi(root, type);

  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");
  assert_throw(ds.size() >= 1, "'dataSource' should contain at least one element");

  type = create_scene_vidi__datatype(ds[0]);

  // load the transfer function as well as the value range
  tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], type);
}

void
create_json_volume_stringify_vidi(const json& root, VolumeDesc& volume)
{
  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");

  // construct file descriptors
  SingleVolumeDesc pv = create_scene_vidi__volume(ds[0]);

  // create multi volume
  volume.dims = pv.dims;
  volume.type = pv.type;
  volume.data.resize(ds.size()); 
  volume.data[0] = pv; // slicing
  for (int i = 1; i < ds.size(); ++i) {
    volume.data.push_back(create_scene_vidi__multivolume(ds[i], pv));
  }

  // load the transfer function as well as the value range
  TransferFunction tfn;
  tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], pv.type);
  volume.range = tfn.range;
}

void
create_json_camera_stringify_vidi(const json& root, Camera& camera)
{
  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");
  SingleVolumeDesc pv = create_scene_vidi__volume(ds[0]);

  camera = create_scene_vidi__camera(root["view"]["camera"]);
  camera.at   -= vec3f(pv.dims) / 2.f;
  camera.from -= vec3f(pv.dims) / 2.f;
}

void
create_json_scene_stringify(const json& root, VolumeDesc& volume, TransferFunction& tfn, Camera& camera)
{
  assert(root.is_object());

  if (root.contains("version")) {
    if (root["version"] == "DIVA") {
      return create_json_scene_diva(root, volume, tfn, camera);
    }
    else if (root["version"] == "VIDI3D") {
      return create_json_scene_vidi(root, volume, tfn, camera);
    }
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_scene_vidi(root, volume, tfn, camera);
}

void
create_json_volume_stringify(const json& root, VolumeDesc& volume)
{
  assert(root.is_object());
  if (root.contains("version")) {
    if      (root["version"] == "DIVA"  ) return create_json_volume_stringify_diva(root, volume);
    else if (root["version"] == "VIDI3D") {}
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_volume_stringify_vidi(root, volume);
}

void
create_json_tfn_stringify(const json& root, TransferFunction& tfn)
{
  assert(root.is_object());
  if (root.contains("version")) {
    if      (root["version"] == "DIVA"  ) return; // TODO 
    else if (root["version"] == "VIDI3D") {}
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_tfn_stringify_vidi(root, tfn);
}

void
create_json_camera_stringify(const json& root, Camera& camera)
{
  assert(root.is_object());
  if (root.contains("version")) {
    if      (root["version"] == "DIVA"  ) return; // TODO 
    else if (root["version"] == "VIDI3D") {}
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_camera_stringify_vidi(root, camera);
}

} // namespace ovr
