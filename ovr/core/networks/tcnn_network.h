#pragma once

#include "nnapi.h"

namespace vnr {

struct TcnnNetwork : AbstractNetwork {
private:
  struct Impl;
  std::unique_ptr<Impl> m;

  int N_NEURONS = -1;
  int N_FEATURES_PER_LEVEL = -1;

  const int INPUT_SIZE;
  const int OUTPUT_SIZE;

public:
  ~TcnnNetwork() override;
  TcnnNetwork(int n_input_dims, int n_output_dims);
  TcnnNetwork(const TcnnNetwork& other) = delete;
  TcnnNetwork(TcnnNetwork&& other) noexcept = default;
  TcnnNetwork& operator=(const TcnnNetwork& other) = delete;
  TcnnNetwork& operator=(TcnnNetwork&& other) noexcept = default;

  int n_input_dims() const override { return INPUT_SIZE; }
  int n_output_dims() const override { return OUTPUT_SIZE; }
  int n_neurons() const override { return N_NEURONS; }
  int n_features_per_level() const override { return N_FEATURES_PER_LEVEL; }
  bool valid() const override;
  size_t get_model_size() const override;
  size_t get_mlp_size() const override;
  size_t get_enc_size() const override;
  size_t training_step() const override;
  double training_loss() const override;
  json serialize_params() const override;
  void deserialize_params(const json& parameters) override;
  json serialize_model() const override;
  void deserialize_model(const json& config) override;
  
  // training API
  void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) override;
  void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const override;
};

}
