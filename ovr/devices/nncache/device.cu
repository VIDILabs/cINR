#include "device.h"
#include "cachemanager.h"
#include <core/instantvnr_types.h>

#define ENABLE_WAVEFRONT_RENDERER

namespace ovr::nncache {

// ------------------------------------------------------------------
// Implementation of the DeviceNNCache
// ------------------------------------------------------------------

void
DeviceNNCache::init(int argc, const char** argv)
{
  if (initialized) throw std::runtime_error("[nncache] device already initialized!");
  initialized = true;

  // --------------------------------------------
  // setup scene
  // --------------------------------------------
  const auto& scene = current_scene;
  auto& sv = parse_single_volume_scene(scene, scene::Volume::STRUCTURED_REGULAR_VOLUME).structured_regular;
  auto& st = scene.instances[0].models[0].volume_model.transfer_function;
  params.tfn.assign([&](TransferFunctionData& d) {
    d.tfn_value_range = st.value_range;
  });

  if (argc >= 5)
    api.lod.scale = std::stof(argv[4]);

  if (argc >= 6) {
    ctls.phi = std::stof(argv[5]);
    ctls.theta = std::stof(argv[6]);
    ctls.intensity = std::stof(argv[7]);
  }

  // --------------------------------------------------------------------------
  // create cache manager
  // --------------------------------------------------------------------------
  vnr::CacheConfig* user_data = (vnr::CacheConfig*)scene.user_data;
  cacheManager = create_cache_manager(  //TODO: refresh with new selected model (pass in id)
    user_data->config, 
    user_data->capacity, 
    user_data->num_levels, 
    sv.data->type
  );

  // --------------------------------------------
  // framebuffer creation
  // --------------------------------------------
  framebuffer.create();
  framebuffer_stream = framebuffer.back_stream();

  // --------------------------------------------
  // create volume texture & transformation
  // --------------------------------------------
  const vec3f scale = sv.grid_spacing * vec3f(sv.data->dims);
  const vec3f translate = sv.grid_origin;
  volume.load_from_array3d_scalar(sv.data); // NOTE: no actually loading the data
  volume.set_space_partition_size(user_data->macrocell_dims, user_data->macrocell_spacings); //TODO: refresh with new selected model
  volume.set_transform(affine3f::translate(translate) * affine3f::scale(scale));
  volume.set_sampling_rate(scene.volume_sampling_rate);
  volume.set_transfer_function(CreateArray1DFloat4CUDA(st.color), CreateArray1DScalarCUDA(st.opacity), st.value_range);
  volume.commit(); // commit volume to make sure we have a valid device handler

  // --------------------------------------------
  // Load precomputed macrocell from file
  // --------------------------------------------
  // NOTE: in NN Cache project, we assume CUDA-style data normalization rule:
  // -- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
  //    about cudaTextureDesc::readMode:
  //       Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted,
  //       regardless of whether or not this cudaTextureDesc::readMode is set cudaReadModeNormalizedFloat is specified.
  range1f mc_range;
  switch (volume.device.volume.type) {
  case VALUE_TYPE_UINT8:
    mc_range.lower = (float)std::numeric_limits<uint8_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<uint8_t>::max();
    break;
  case VALUE_TYPE_INT8:
    mc_range.lower = (float)std::numeric_limits<int8_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<int8_t>::max();
    break;
  case VALUE_TYPE_UINT16:
    mc_range.lower = (float)std::numeric_limits<uint16_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<uint16_t>::max();
    break;
  case VALUE_TYPE_INT16:
    mc_range.lower = (float)std::numeric_limits<int16_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<int16_t>::max();
    break;
  default: break;
  }

  std::cout << "[mc] range " << mc_range << std::endl;
  set_space_partition(volume.space_partition, user_data->macrocell, mc_range, framebuffer_stream);

  // --------------------------------------------
  // create wavefront renderer
  // --------------------------------------------
  const auto& data = volume.device.volume;
  const auto& iter = volume.device.sp;
  api.stream = framebuffer_stream;
  api.init(volume.device.transform, 
    data.type, data.dims, 
    // vnr::range1f(data.lower.v, data.upper.v),
    iter.dims, iter.spac,
    (vnr::vec2f*)iter.value_ranges,
    iter.majorants
    // user_data->macrocell
  );

  // --------------------------------------------------------------------------
  // call commit
  // --------------------------------------------------------------------------  
  commit();
}

void
DeviceNNCache::swap()
{
  framebuffer.safe_swap();
  framebuffer_stream = framebuffer.back_stream();
  api.stream = framebuffer_stream;
}

void 
DeviceNNCache::commit_material() 
{
  if (check(ctls.ambient))   { lp.mat_scivis.ambient   = ctls.ambient.ref(); }
  if (check(ctls.diffuse))   { lp.mat_scivis.diffuse   = ctls.diffuse.ref(); }
  if (check(ctls.specular))  { lp.mat_scivis.specular  = ctls.specular.ref(); }
  if (check(ctls.shininess)) { lp.mat_scivis.shininess = ctls.shininess.ref(); }
}

void 
DeviceNNCache::commit_lighting() 
{
  if (check(ctls.phi) || check(ctls.theta)) {
    const float phi_rad   = ctls.phi.get()   * (M_PI/180);
    const float theta_rad = ctls.theta.get() * (M_PI/180);
    const float radius = 2415.8;
    lp.l_distant.direction = vec3f( 
      radius * sin(phi_rad) * cos(theta_rad) * (180/M_PI),
      radius * sin(phi_rad) * sin(theta_rad) * (180/M_PI),
      radius * cos(phi_rad) * (180/M_PI)
    );
  }
  if (check(ctls.intensity)) {
    lp.l_distant.color = vec3f(ctls.intensity.get());
  }
}

void
DeviceNNCache::commit()
{
  if (check(params.fbsize)) {
    lp.frame.size = params.fbsize.ref();
    CUDA_SYNC_CHECK(); /* sync rendering */
    framebuffer.resize(params.fbsize.ref());
  }

  // camera parameters
  if (check(params.camera)) { 
    camera = params.camera.ref(); 
  }

  // volume parameters
  if (check(params.tfn)) {
    const auto& tfn = params.tfn.ref();
    volume.set_transfer_function(tfn.tfn_colors, tfn.tfn_alphas, tfn.tfn_value_range);
    volume_changed = true;
  }
  if (check(params.volume_sampling_rate)) {
    volume.set_sampling_rate(params.volume_sampling_rate.get());
    volume_changed = true;
  }
  if (check(params.volume_density_scale)) {
    volume.set_density_scale(params.volume_density_scale.get());
    volume_changed = true;
  }
  if (volume_changed) {
    volume.commit();
    volume_changed = false;
  }

  // light & material parameters
  commit_material();
  commit_lighting();

  // other parameters
  if (check(params.path_tracing)) {
    lp.enable_path_tracing = params.path_tracing.ref();
    rendering_mode = lp.enable_path_tracing ? 3 : 0;
  }
  if (check(params.sparse_sampling)) {
    lp.enable_sparse_sampling = params.sparse_sampling.ref();
  }
  if (check(params.frame_accumulation)) {
    lp.enable_frame_accumulation = params.frame_accumulation.ref();
  }
  if (check(ctls.wavefront))     { waverfront = ctls.wavefront.ref(); }
  if (check(ctls.cachemode))     { api.cachemode = ctls.cachemode.ref(); }
  if (check(ctls.lod_scale))     { api.lod.scale = ctls.lod_scale.ref(); }
  if (check(ctls.lod_threshold)) { api.lod.threshold = ctls.lod_threshold.ref(); }

  // finalize
  if (dirty) {
    api.update(rendering_mode, volume.device.tfn, 
      volume.sampling_rate, volume.density_scale,
      vec3f(0), vec3f(1), to_vnr(camera), lp.frame.size
    );
  }
}

void
DeviceNNCache::render()
{
  if (lp.frame.size.x <= 0 || lp.frame.size.y <= 0) return;
  lp.frame.rgba = (vec4f*)framebuffer.back_dpointer(/*layout=*/0);

#ifdef ENABLE_WAVEFRONT_RENDERER
  if (waverfront) 
  {
    api.render((vec4f*)framebuffer.back_dpointer(0), nullptr, *cacheManager);
  }
  else
#endif
  {
    ++lp.frame_index;
    api.megakernel.render(framebuffer_stream, lp, volume.device, *cacheManager, dirty);
  }

  /* post rendering */
  variance = 0.f; // TODO properly calculate variance
  dirty = false;
}

void
DeviceNNCache::mapframe(FrameBufferData* fb)
{
  // CUDA_CHECK(cudaStreamSynchronize(framebuffer_stream));
  const size_t num_bytes = framebuffer.size().long_product();
  fb->rgba->set_data(framebuffer.front_dpointer(0), num_bytes * sizeof(vec4f), CrossDeviceBuffer::DEVICE_CUDA);
  fb->size = framebuffer.size();
}

}

static bool file_exists_test(std::string name) { std::ifstream f(name.c_str()); return f.good(); }
static bool file_exists_test(std::string name, const std::string& dir, std::string& out) {
  if (file_exists_test(name)) { out = name; return true; }
  else if (!dir.empty()) {
    if      (file_exists_test(dir + "/"  + name)) { out = name; return true; }
    else if (file_exists_test(dir + "\\" + name)) { out = name; return true; }
  }
  return false;
}
static std::string valid_filename(const vnr::json& in, std::string dir, const std::string& key) {
  std::string file;
  if (!in.contains(key)) { throw std::runtime_error("JSON key '" + key + "' doesnot exist"); }
  const auto& js = in[key];
  if (js.is_array()) {
    for (auto& s : js) { if (file_exists_test(s.get<std::string>(), dir, file)) { return file; } }
    throw std::runtime_error("Cannot find file for '" + key + "'.");
  }
  else {
    if (file_exists_test(js.get<std::string>(), dir, file)) { return file; }
    throw std::runtime_error("File '" + js.get<std::string>() + "' does not exist.");
  }
}

OVR_REGISTER_OBJECT(ovr::MainRenderer, renderer, ovr::nncache::DeviceNNCache, nncache)

OVR_REGISTER_SCENE_LOADER(nncache, filename)
{
  std::ifstream file(filename);
  vnr::json root = vnr::json::parse(file, nullptr, true, true);

  // TODO: Make sure we have version control in the JSON file
  // if (!root.contains("version") || root["version"] != "nncache-1.0") {
  //   throw std::runtime_error("unsupported nncache version");
  // }

  vnr::TransferFunction tfn;
  vnr::Camera camera;
  vnr::vec3i dims; 
  vnr::ValueType type;
  vnr::create_json_scene_stringify(root, tfn, camera, dims, type);

  // ------------------------------------------------------------------
  // create scene
  // ------------------------------------------------------------------
  using namespace ovr;
  using namespace ovr::scene;

  Scene scene;

  Instance instance;
  instance.transform = affine3f::translate(vec3f(0));

  Volume volume{};
  volume.type = ovr::scene::Volume::STRUCTURED_REGULAR_VOLUME;
  volume.structured_regular.data = std::make_shared<Array<3>>();
  volume.structured_regular.data->dims = dims;
  volume.structured_regular.data->type = nncache::to_ovr(type);

  ovr::scene::TransferFunction transfer_function{};

  std::vector<vec4f> color(tfn.color.size());
  for (int i = 0; i < tfn.color.size(); ++i) {
    color[i] = vec4f(tfn.color[i], 1.f);
  }

  std::vector<float> alpha(tfn.alpha.size());
  for (int i = 0; i < tfn.alpha.size(); ++i) {
    alpha[i] = tfn.alpha[i].y;
  }

  transfer_function.color   = ovr::CreateArray1DFloat4(color);
  transfer_function.opacity = ovr::CreateArray1DScalar(alpha);
  transfer_function.value_range.x = tfn.range.lo;
  transfer_function.value_range.y = tfn.range.hi;

  Texture texture;
  texture.type = Texture::VOLUME_TEXTURE;
  texture.volume.volume = volume;
  scene.textures.push_back(texture);
  
  Model model;
  model.type = Model::VOLUMETRIC_MODEL;
  model.volume_model.volume_texture = scene.textures.size() - 1;
  model.volume_model.transfer_function = transfer_function;

  instance.models.push_back(model);
  scene.instances.push_back(instance);
  scene.camera = nncache::to_ovr(camera);

  vnr::CacheConfig* user_data = new vnr::CacheConfig();

  // ------------------------------------------------------------------
  // parse cache configuration
  // ------------------------------------------------------------------
  if (!root.contains("cache")) { throw std::runtime_error("JSON missing cache configuration"); }
  const auto& cache = root["cache"];

  if (cache.contains("config")) {
    user_data->config = valid_filename(cache, "", "config");
  }
  else {
    throw std::runtime_error("JSON cache missing configuration filename");
  }

  if (cache.contains("capacity")) {
    user_data->capacity = vnr::vec3i(
      cache["capacity"]["x"].get<int>(), 
      cache["capacity"]["y"].get<int>(), 
      cache["capacity"]["z"].get<int>()
    );
  }

  if (cache.contains("numLevels")) {
    user_data->num_levels = cache["numLevels"].get<int>();
  }

  // ------------------------------------------------------------------
  // parse macrocell configuration
  // ------------------------------------------------------------------

  if (!root.contains("macrocell")) { throw std::runtime_error("JSON missing macrocell configuration"); }
  const auto& mc = root["macrocell"];

  if (mc.contains("fileName")) {
    user_data->macrocell = valid_filename(mc, "", "fileName");

    vnrJson params;
    vnrLoadJsonBinary(params, user_data->macrocell);

    if (params.contains("macrocell")) {
        user_data->macrocell_dims = vec3i(
          params["macrocell"]["dims"]["x"].get<int>(),
          params["macrocell"]["dims"]["y"].get<int>(),
          params["macrocell"]["dims"]["z"].get<int>());
        
        user_data->macrocell_spacings = vec3f(
          params["macrocell"]["spacings"]["x"].get<float>(),
          params["macrocell"]["spacings"]["y"].get<float>(),
          params["macrocell"]["spacings"]["z"].get<float>());
        
    }
      else 
      throw std::runtime_error("Failed to open Macrocell data in " + user_data->macrocell);
  }


  // Compute macrocell on the fly
  // if (user_data->macrocell.empty()) {
  //   vnr::vec3i mc_size;
  //   if (mc.contains("mcSize")) {
  //     user_data->macrocell_size = mc_size = vnr::vec3i(
  //       mc["mcSize"]["x"].get<int>(), 
  //       mc["mcSize"]["y"].get<int>(), 
  //       mc["mcSize"]["z"].get<int>()
  //     );
  //   }
  //   else {
  //     throw std::runtime_error("JSON macrocell missing dimensions");
  //   }
  //   std::string volume_path = ovr::nncache::get_cache_datapath(user_data->config);
  //   std::string output_path = "";
  //   tdns::Macrocell macrocell(mc_size, dims, type);
  //   macrocell.process(volume_path, output_path);
  //   user_data->macrocell = output_path;
  // }

  // ------------------------------------------------------------------
  // write output
  // ------------------------------------------------------------------
  scene.user_data = user_data;

  return scene;
}                            
