//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "api_internal.h"

#include "ovr/devices/nncache/framebuffer.h"
#include "ovr/devices/nncache/renderer/renderer.h"

namespace vnr {

using ovr::nncache::RenderObject;
using FrameBuffer = ovr::nncache::DoubleBufferObject<vec4f>;

struct RenderContext
{
  vnrVolume volume;
  Camera camera;

  TransferFunctionObject tfn;
  RenderObject render;

  // other states
  int rendering_mode{ VNR_INVALID };

  // volume states
  float sampling_rate{ 1.f };
  float density_scale{ 1.f };

  // framebuffer states
  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  vec2i framebuffer_size;
  bool framebuffer_reset{ true };
};

}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

using namespace vnr;

static ovr::ValueType to_ovr(ValueType type) {
  switch (type) {
  case VALUE_TYPE_UINT8:  return ovr::ValueType::VALUE_TYPE_UINT8;
  case VALUE_TYPE_INT8:   return ovr::ValueType::VALUE_TYPE_INT8;
  case VALUE_TYPE_UINT16: return ovr::ValueType::VALUE_TYPE_UINT16;
  case VALUE_TYPE_INT16:  return ovr::ValueType::VALUE_TYPE_INT16;
  case VALUE_TYPE_UINT32: return ovr::ValueType::VALUE_TYPE_UINT32;
  case VALUE_TYPE_INT32:  return ovr::ValueType::VALUE_TYPE_INT32;
  case VALUE_TYPE_FLOAT:  return ovr::ValueType::VALUE_TYPE_FLOAT;
  case VALUE_TYPE_DOUBLE: return ovr::ValueType::VALUE_TYPE_DOUBLE;
  case VALUE_TYPE_FLOAT2: return ovr::ValueType::VALUE_TYPE_FLOAT2;
  case VALUE_TYPE_FLOAT3: return ovr::ValueType::VALUE_TYPE_FLOAT3;
  case VALUE_TYPE_FLOAT4: return ovr::ValueType::VALUE_TYPE_FLOAT4;
  default: throw std::runtime_error("unknown type encountered");
  }
}

static ovr::nncache::DeviceTransferFunction to_ovr(const DeviceTransferFunction& tfn) {
  ovr::nncache::DeviceTransferFunction ret;
  ret.alphas.type   = to_ovr(tfn.alphas.type);
  ret.alphas.dims.v = tfn.alphas.length;
  ret.alphas.data   = tfn.alphas.data;
  ret.alphas.rawptr = tfn.alphas.rawptr;
  ret.colors.type   = to_ovr(tfn.colors.type);
  ret.colors.dims.v = tfn.colors.length;
  ret.colors.data   = tfn.colors.data;
  ret.colors.rawptr = tfn.colors.rawptr;
  ret.value_range = vnr::range1f(tfn.range.lo, tfn.range.hi);
  ret.range_rcp_norm = 1.f / ret.value_range.span();
  return ret;
}

void vnrLoadJsonText(vnrJson& output, std::string filename)
{
  std::ifstream file(filename);
  output = vnr::json::parse(file, nullptr, true, true);
}

void vnrLoadJsonBinary(vnrJson& output, std::string filename)
{
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (file.read(buffer.data(), size)) {
    output = vnr::json::from_bson(buffer);
  }
}

void vnrSaveJsonText(const vnrJson& root, std::string filename)
{
  std::ofstream ofs(filename, std::ios::out);
  ofs << std::setw(4) << root << std::endl;
  ofs.close();
}

void vnrSaveJsonBinary(const vnrJson& root, std::string filename)
{
  const auto broot = json::to_bson(root);
  std::ofstream ofs(filename, std::ios::binary | std::ios::out);
  ofs.write((char*)broot.data(), broot.size());
  ofs.close();
}


vnrJson vnrCreateJsonText(std::string filename)
{
  vnr::json output;
  vnrLoadJsonText(output, filename);
  return output;
}

vnrJson vnrCreateJsonBinary(std::string filename)
{
  vnr::json output;
  vnrLoadJsonBinary(output, filename);
  return output;
}


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrCamera vnrCreateCamera()
{
  return std::make_shared<Camera>();
}

vnrCamera vnrCreateCamera(const vnrJson& scene)
{
  auto cam = std::make_shared<Camera>();
  if (scene.is_string()) {
    create_json_camera(scene.get<std::string>(), *cam);
  }
  else {
    create_json_camera_stringify(scene, *cam);
  }
  return cam;
}

void vnrCameraSet(vnrCamera self, vnr::vec3f from, vnr::vec3f at, vnr::vec3f up)
{
  *self = { 
    /*from*/ from,
    /* at */ at,
    /* up */ up 
  };
}

void vnrCameraSet(vnrCamera self, const vnrJson& scene)
{
  // auto cam = std::make_shared<Camera>();
  if (scene.is_string()) {
    create_json_camera(scene.get<std::string>(), *self);
  }
  else {
    create_json_camera_stringify(scene, *self);
  }
}

vnr::vec3f vnrCameraGetPosition(vnrCamera self)
{
  return self->from;
}

vnr::vec3f vnrCameraGetFocus(vnrCamera self)
{
  return self->at;
}

vnr::vec3f vnrCameraGetUpVec(vnrCamera self)
{
  return self->up;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

inline std::shared_ptr<SimpleVolumeContext> 
castSimpleVolume(vnrVolume self) 
{
  if (self->isNetwork()) {
    throw std::runtime_error("expecting a simple volume");
  }
  return std::dynamic_pointer_cast<SimpleVolumeContext>(self);
}

inline std::shared_ptr<NeuralVolumeContext> 
castNeuralVolume(vnrVolume self) 
{
  if (!self->isNetwork()) {
    throw std::runtime_error("expecting a neural volume");
  }
  return std::dynamic_pointer_cast<NeuralVolumeContext>(self);
}

// simple volume

vnrVolume vnrCreateSimpleVolume(const void* data, vnr::vec3i dims, std::string type, vnr::range1f range, std::string sampling_mode)
{
  auto ret = std::make_shared<SimpleVolumeContext>();
  ret->dims = dims;
  ret->type = vnr::value_type(type);
  ret->range = range;
  ret->source.load(data, dims, type, range, sampling_mode);
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

vnrVolume vnrCreateSimpleVolume(const vnrJson& scene, std::string sampling_mode, bool save_loaded_volume)
{
  auto ret = std::make_shared<SimpleVolumeContext>();
  VolumeDesc desc;
  if (scene.is_string()) {
    create_json_volume(scene.get<std::string>(), desc);
  }
  else {
    create_json_volume_stringify(scene, desc);
  }
  ret->dims = desc.dims;
  ret->type = desc.type;
  ret->range = desc.range;
  ret->source.load(desc, sampling_mode, save_loaded_volume);
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

void vnrSimpleVolumeSetCurrentTimeStep(vnrVolume self, int time)
{
  auto sv = castSimpleVolume(self);
  sv->source.set_current_timestep(time);
}

int vnrSimpleVolumeGetNumberOfTimeSteps(vnrVolume self)
{
  auto sv = castSimpleVolume(self);
  return sv->source.get_num_timesteps();
}

// neural volume

vnrVolume vnrCreateNeuralVolume(const json& config, vnrVolume groundtruth, bool online_macrocell_construction, size_t batchsize)
{
  auto& source = castSimpleVolume(groundtruth)->source;
  auto ret = std::make_shared<NeuralVolumeContext>(batchsize);
  ret->dims = groundtruth->dims;
  ret->type = groundtruth->type;
  ret->range = groundtruth->range;
  if (config.is_string()) {
    ret->neural.set_network(ret->dims, config.get<std::string>(), &source, !online_macrocell_construction);
  }
  else {
    ret->neural.set_network_from_json(ret->dims, config, &source, !online_macrocell_construction);
  }
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

vnrVolume vnrCreateNeuralVolume(const json& config, vnr::vec3i dims, size_t batchsize)
{
  auto ret = std::make_shared<NeuralVolumeContext>(batchsize);
  ret->dims = dims;
  ret->type = vnr::VALUE_TYPE_FLOAT;
  ret->range = range1f(0, 1);
  if (config.is_string()) {
    ret->neural.set_network(ret->dims, config.get<std::string>(), nullptr, false);
  }
  else {
    ret->neural.set_network_from_json(ret->dims, config, nullptr, false);
  }
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

vnrVolume vnrCreateNeuralVolume(const json& params, size_t batchsize)
{
  vec3i dims;
  if (params.contains("volume")) {
    dims.x = params["volume"]["dims"]["x"].get<int>();
    dims.y = params["volume"]["dims"]["y"].get<int>();
    dims.z = params["volume"]["dims"]["z"].get<int>();
  }
  else {
    throw std::runtime_error("expecting a model config with volume dims tag");
  }
  auto ret = vnrCreateNeuralVolume(params["model"], dims, batchsize);
  vnrNeuralVolumeSetParams(ret, params);
  return ret;
}

void vnrNeuralVolumeTrain(vnrVolume self, int steps, bool fast_mode, bool verbose)
{
  auto nv = castNeuralVolume(self);
  nv->neural.train(steps, fast_mode, verbose);
}

void vnrNeuralVolumeDecodeProgressive(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  nv->neural.decode_progressive();
}

void vnrNeuralVolumeDecode(vnrVolume self, float* output)
{
  auto nv = castNeuralVolume(self);
  nv->neural.decode_volume(output, nv->neural.get_data_dims());
}

void vnrNeuralVolumeDecodeInference(vnrVolume self, std::string filename)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_inference_volume(filename, nv->neural.get_data_dims());
}

void vnrNeuralVolumeDecodeReference(vnrVolume self, std::string filename)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_reference_volume(filename, nv->neural.get_data_dims());
}

void vnrNeuralVolumeSerializeParams(vnrVolume self, std::string filename)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_params(filename);
}

void vnrNeuralVolumeSerializeParams(vnrVolume self, vnrJson& params)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_params_to_json(params);
}

void vnrNeuralVolumeSetModel(vnrVolume self, const vnrJson& config)
{
  auto nv = castNeuralVolume(self);
  if (config.is_string()) {
    nv->neural.set_network(config.get<std::string>());
  }
  else {
    nv->neural.set_network_from_json(config);
  }
}

void vnrNeuralVolumeSetParams(vnrVolume self, const vnr::json& params)
{
  auto nv = castNeuralVolume(self);
  if (params.is_string()) {
    nv->neural.load_params(params.get<std::string>());
  }
  else {
    nv->neural.load_params_from_json(params);
  }
}

double vnrNeuralVolumeGetMSE(vnrVolume self, bool verbose)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_mse(self->dims, !verbose);
}

double vnrNeuralVolumeGetPSNR(vnrVolume self, bool verbose)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_psnr(self->dims, !verbose);
}

double vnrNeuralVolumeGetSSIM(vnrVolume self, bool verbose)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_mssim(self->dims, !verbose);
}

double vnrNeuralVolumeGetTestingLoss(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  float loss;
  nv->neural.test(&loss);
  return loss;
}

double vnrNeuralVolumeGetTrainingLoss(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  NeuralVolume::Statistics stats;
  nv->neural.statistics(stats);
  return stats.loss;
}

int vnrNeuralVolumeGetTrainingStep(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  NeuralVolume::Statistics stats;
  nv->neural.statistics(stats);
  return (int)stats.step;
}

int vnrNeuralVolumeGetNumberOfBlobs(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_num_blobs();
}

int vnrNeuralVolumeGetNBytesMultilayerPerceptron(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_mlp_size();
}

int vnrNeuralVolumeGetNBytesEncoding(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_enc_size();
}

// general

void vnrVolumeSetClippingBox(vnrVolume self, vnr::vec3f lower, vnr::vec3f upper)
{
  vnr::affine3f transform;
  if (self->isNetwork()) {
    transform = std::dynamic_pointer_cast<NeuralVolumeContext>(self)->neural.get_data_transform();
  }
  else {
    transform = std::dynamic_pointer_cast<SimpleVolumeContext>(self)->source.get_data_transform();
  }

  lower -= vec3f(self->dims)/2;
  upper -= vec3f(self->dims)/2;
  lower = gdt::xfmPoint(transform.inverse(), lower);
  upper = gdt::xfmPoint(transform.inverse(), upper);
  self->clipbox.lower = lower;
  self->clipbox.upper = upper;
}

void vnrVolumeSetScaling(vnrVolume self, vnr::vec3f scale)
{
  VolumeObject* v = nullptr;
  if (self->isNetwork()) {
    v = &(std::dynamic_pointer_cast<NeuralVolumeContext>(self)->neural);
  }
  else {
    v = &(std::dynamic_pointer_cast<SimpleVolumeContext>(self)->source);
  }
  vnr::affine3f transform = vnr::affine3f::scale(scale) * v->get_data_transform();
  v->set_data_transform(transform);
}

// vnr::range1f vnrVolumeGetValueRange(vnrVolume self)
// {
//   if (self->isNetwork()) {
//     return std::dynamic_pointer_cast<NeuralVolumeContext>(self)->neural.get_data_value_range();
//   }
//   else {
//     return std::dynamic_pointer_cast<SimpleVolumeContext>(self)->source.get_data_value_range();
//   }
// }

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrTransferFunction vnrCreateTransferFunction()
{
  return std::make_shared<TransferFunction>();
}

vnrTransferFunction vnrCreateTransferFunction(const vnr::json& scene)
{
  auto tfn = std::make_shared<TransferFunction>();
  if (scene.is_string()) {
    create_json_tfn(scene.get<std::string>(), *tfn);
  }
  else {
    create_json_tfn_stringify(scene, *tfn);
  }
  return tfn;
}

void vnrTransferFunctionSetColor(vnrTransferFunction tfn, const std::vector<vnr::vec3f>& colors)
{
  tfn->color = colors;
}

void vnrTransferFunctionSetAlpha(vnrTransferFunction tfn, const std::vector<vnr::vec2f>& alphas)
{
  tfn->alpha = alphas;
}

void vnrTransferFunctionSetValueRange(vnrTransferFunction tfn, vnr::range1f range)
{
  tfn->range = range;
}

const std::vector<vnr::vec3f>& vnrTransferFunctionGetColor(vnrTransferFunction tfn)
{
  return tfn->color;
}

const std::vector<vnr::vec2f>& vnrTransferFunctionGetAlpha(vnrTransferFunction tfn)
{
  return tfn->alpha;
}

const vnr::range1f& vnrTransferFunctionGetValueRange(vnrTransferFunction tfn)
{
  return tfn->range;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrRenderer vnrCreateRenderer(vnrVolume v)
{
  auto self = std::make_shared<RenderContext>();
  self->volume = v;
  self->framebuffer.create(/*async=*/true);
  self->framebuffer_stream = self->framebuffer.back_stream();
  self->render.stream = self->framebuffer_stream;
  if (self->volume->isNetwork()) {
    auto& source = std::dynamic_pointer_cast<NeuralVolumeContext>(self->volume)->neural;
    self->render.init(
      source.get_data_transform(),
      to_ovr(source.get_data_type()), 
      source.get_data_dims(), 
      // source.get_data_value_range(),
      source.get_macrocell_dims(), 
      source.get_macrocell_spacings(), 
      source.get_macrocell_value_range(), 
      source.get_macrocell_max_opacity()
    );
  }
  else {
    auto& source = std::dynamic_pointer_cast<SimpleVolumeContext>(self->volume)->source;
    // std::cout << "REF MC " <<  source.get_macrocell_value_range() << std::endl;
    self->render.init(
      source.get_data_transform(),
      to_ovr(source.get_data_type()), 
      source.get_data_dims(), 
      // source.get_data_value_range(),
      source.get_macrocell_dims(), 
      source.get_macrocell_spacings(), 
      source.get_macrocell_value_range(), 
      source.get_macrocell_max_opacity()
    );
  }
  self->framebuffer_reset = true;
  return self;
}

void vnrRendererSetMode(vnrRenderer self, int mode)
{
  self->rendering_mode = mode;
  self->framebuffer_reset = true;
}

void vnrRendererSetDenoiser(vnrRenderer self, bool flag)
{
  ;
}

void vnrRendererSetVolumeSamplingRate(vnrRenderer self, float rate)
{
  self->sampling_rate = rate;
  self->framebuffer_reset = true;
}

void vnrRendererSetVolumeDensityScale(vnrRenderer self, float value)
{
  self->density_scale = value;
  self->framebuffer_reset = true;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void vnrRendererSetTransferFunction(vnrRenderer self, vnrTransferFunction _tfn)
{
  // range1f original_data_range;
  auto& tfn = *_tfn;
  if (self->volume->isNetwork()) {
    auto& source = std::dynamic_pointer_cast<NeuralVolumeContext>(self->volume)->neural;
    source.set_transfer_function(tfn.color, tfn.alpha, tfn.range);
    // original_data_range = source.get_data_value_range();
  }
  else {
    auto& source = std::dynamic_pointer_cast<SimpleVolumeContext>(self->volume)->source;
    source.set_transfer_function(tfn.color, tfn.alpha, tfn.range);
    // original_data_range = source.get_data_value_range();
  }

  // self->renderer.set_transfer_function(tfn.color, tfn.alpha, tfn.range);
  self->tfn.update(tfn, /*original_data_range,*/ self->framebuffer_stream);
  self->framebuffer_reset = true;
}

void vnrRendererSetCamera(vnrRenderer self, vnrCamera cam)
{
  self->camera = *cam;
  self->framebuffer_reset = true;
}

void vnrRendererSetFramebufferSize(vnrRenderer self, vec2i fbsize)
{
  self->framebuffer.resize(fbsize);
  self->framebuffer_size = fbsize;
  self->framebuffer_reset = true;
}

vnr::vec4f *vnrRendererMapFrame(vnrRenderer self)
{
  self->framebuffer.safe_swap();

  // NOTE The followings happen concurrent with rendering (ideally at least)

  self->framebuffer_stream = self->framebuffer.back_stream();
  self->render.stream = self->framebuffer_stream;
  // CUDA_CHECK(cudaStreamSynchronize(self->framebuffer_stream));
  if (self->framebuffer_size.long_product() != 0) { 
    self->framebuffer.download_front();
  }
  vec4f *pixels = (vec4f*)self->framebuffer.front_hpointer();

  return pixels;
}

void vnrRendererResetAccumulation(vnrRenderer self)
{
  self->framebuffer_reset = true;
}

void vnrRender(vnrRenderer self)
{
  if (self->framebuffer_reset) {
    self->render.update(
      self->rendering_mode, 
      to_ovr(self->tfn.tfn),
      self->sampling_rate,
      self->density_scale,
      self->volume->clipbox.lower,
      self->volume->clipbox.upper,
      self->camera,
      self->framebuffer_size
    );
    self->framebuffer_reset = false;
  }

  if (self->volume->isNetwork()) {
    auto& source = std::dynamic_pointer_cast<NeuralVolumeContext>(self->volume)->neural;    
    self->render.render((vec4f*)self->framebuffer.back_dpointer(), &source, source.texture());
  }
  else {
    auto& source = std::dynamic_pointer_cast<SimpleVolumeContext>(self->volume)->source;
    self->render.render((vec4f*)self->framebuffer.back_dpointer(), nullptr, source.texture());
  }
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void vnrResetMaxMemory()
{
  util::max_nbytes_allocated() = 0;
}

void vnrMemoryQuery(size_t* used_by_self, size_t* used_by_tcnn, size_t* used_peak, size_t* used_total)
{
  if (used_by_self) *used_by_self = util::tot_nbytes_allocated();
  if (used_by_tcnn) *used_by_tcnn = NeuralVolume::tot_nbytes_allocated_by_tcnn();
  if (used_peak) *used_peak = util::max_nbytes_allocated() + NeuralVolume::max_nbytes_allocated_by_tcnn();
  if (used_total) {
    unsigned long long tmp; util::getUsedGPUMemory(&tmp); *used_total = tmp;
  }
}

void vnrMemoryQueryPrint(const char* str)
{
  size_t used_by_self;
  size_t used_by_tcnn;
  size_t used_peak;
  size_t used_total;
  vnrMemoryQuery(&used_by_self, &used_by_tcnn, &used_peak, &used_total);
  printf("%s: total used %s, self %s, tcnn %s, unknown %s, peak %s\n", str,
         util::prettyBytes(used_total).c_str(),
         util::prettyBytes(used_by_self).c_str(),
         util::prettyBytes(used_by_tcnn).c_str(),
         util::prettyBytes(used_total - used_by_self - used_by_tcnn).c_str(),
         util::prettyBytes(used_peak).c_str()
  );
}

void vnrFreeTemporaryGPUMemory()
{
  NeuralVolume::free_temporary_gpu_memory_by_tcnn();
}

void vnrCompilationStatus(const char* str)
{
  printf("%s: Instant VNR Summary\n", str);

#if ADAPTIVE_SAMPLING
  printf("    macrocell acceleration: enabled\n");
  printf("    macrocell size: %d\n", 1 << MACROCELL_SIZE_MIP);
#else
  printf("    macrocell acceleration: disabled\n");
#endif


#ifdef ENABLE_OUT_OF_CORE
  printf("    out-of-core sampler: enabled\n");
#else
  printf("    out-of-core sampler: disabled\n");
#endif
}
