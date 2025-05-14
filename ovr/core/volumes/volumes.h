#pragma once

#include "macrocell.h"
#include "../samplers/sampler.h"
#include "../instantvnr_types.h"

#include <json/json.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <tuple>
#include <string>
#include <memory>

/* Volumetric Neural Representation */
namespace vnr {

using json = nlohmann::json;

struct SimpleVolume : VolumeObject {
private:
  VolumeDesc desc;
  cudaTextureObject_t tex;
  TransferFunctionObject tfn;

public:
  Sampler sampler;
  MacroCell macrocell;
  std::string mode;

  void load(const void* data, vec3i dims, std::string type, range1f range, std::string sampling_mode);
  void load(const VolumeDesc& descriptor, std::string sampling_mode, bool save_volume = false);

  uint32_t get_num_timesteps() const { return (uint32_t)desc.data.size(); }
  void set_current_timestep(int index);

  // common API
  const cudaTextureObject_t& texture()  const override { return tex; }
  ValueType get_data_type()             const override { return sampler->type(); }
  // range1f   get_data_value_range()      const override { return range1f(sampler->lower(), sampler->upper()); }
  vec3i     get_data_dims()             const override { return sampler->rendering_dims(); }
  affine3f  get_data_transform()        const override { return sampler->transform(); }
  float*    get_macrocell_max_opacity() const override { return macrocell.d_max_opacity(); }
  vec2f*    get_macrocell_value_range() const override { return macrocell.d_value_range(); }
  vec3i     get_macrocell_dims()        const override { return macrocell.dims(); }
  vec3f     get_macrocell_spacings()    const override { return macrocell.spacings(); }
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) override;
  void set_data_transform(affine3f transform) override;
};

class NeuralVolume : public VolumeObject
{
public:
  static size_t tot_nbytes_allocated_by_tcnn();
  static size_t max_nbytes_allocated_by_tcnn();
  static void   free_temporary_gpu_memory_by_tcnn();

  struct Statistics {
    size_t step;
    double loss;
  };

  ~NeuralVolume();
  NeuralVolume(size_t batchsize = 1 << 16);
  NeuralVolume(const NeuralVolume& other) = delete;
  NeuralVolume(NeuralVolume&& other) noexcept = default;
  NeuralVolume& operator=(const NeuralVolume& other) = delete;
  NeuralVolume& operator=(NeuralVolume&& other) noexcept = default;

  // common API
  const cudaTextureObject_t& texture()  const override;
  ValueType get_data_type() const override;
  // range1f   get_data_value_range() const override;
  vec3i     get_data_dims() const override;
  affine3f  get_data_transform() const override;
  float*    get_macrocell_max_opacity() const override;
  vec2f*    get_macrocell_value_range() const override;
  vec3i     get_macrocell_dims()        const override;
  vec3f     get_macrocell_spacings()    const override;

  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) override;
  void set_data_transform(affine3f transform) override;

  // Private API

  void set_network(vec3i dims, std::string config_filename,  SimpleVolume* reference, bool use_reference_macrocell);
  void set_network_from_json(vec3i dims, const json& config, SimpleVolume* reference, bool use_reference_macrocell);

  void set_network(std::string config_filename);
  void set_network_from_json(const json& config);

  uint32_t  get_num_blobs() const;

  uint32_t get_mlp_size() const;
  uint32_t get_enc_size() const;

  float get_mse(vec3i resolution, bool quiet = false) const;
  float get_psnr(vec3i resolution, bool quiet = false) const;
  float get_mssim(vec3i resolution, bool quiet = false) const;
  vec2f get_macrocell_psnr() const;

  // TODO add more functions
  int get_network_width() const;
  int get_network_features_per_level() const;
  // void* get_network() const; // access the tcnn network in terms of raw pointer

  void decode_volume(float* output, vec3i resolution) const;
  void save_reference_volume(std::string filename, vec3i resolution) const;
  void save_inference_volume(std::string filename, vec3i resolution) const;

  void save_params(std::string filename) const;
  void save_params_to_json(json& params) const;

  void load_params(std::string filename);
  void load_params_from_json(const json& params);

  // trigger a training step
  void train(size_t steps, bool fast_mode = false, bool verbose = false);

  // inference the tcnn network
  void inference(int len, const float* d_input, float* d_output, cudaStream_t stream);

  // trigger an inference step
  void decode_progressive();

  // trigger a testing step
  void test(float* loss);

  // get current training statistics
  void statistics(Statistics& stats);

public:
  struct Impl;
  std::unique_ptr<Impl> pimpl; // pointer to the internal implementation
};

}
