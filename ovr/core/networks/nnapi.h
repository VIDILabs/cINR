#pragma once


#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <json/json.hpp>

#include "../instantvnr_types.h"

#include <memory>

namespace vnr {

// ------------------------------------------------------------------
// Shared Definitions
// ------------------------------------------------------------------

using json = nlohmann::json;

using TCNN_NAMESPACE :: GPUMatrix;
using TCNN_NAMESPACE :: GPUMemory;
using TCNN_NAMESPACE :: GPUMatrixDynamic;

using GPUColumnMatrix = TCNN_NAMESPACE :: GPUMatrix<float, TCNN_NAMESPACE :: MatrixLayout::ColumnMajor>;
using GPURowMatrix    = TCNN_NAMESPACE :: GPUMatrix<float, TCNN_NAMESPACE :: MatrixLayout::RowMajor>;

using TCNN_NAMESPACE :: json_binary_to_gpu_memory;
using TCNN_NAMESPACE :: gpu_memory_to_json_binary;

// ------------------------------------------------------------------
// Public Interface
// ------------------------------------------------------------------

struct AbstractNetwork {
  virtual ~AbstractNetwork() {}
  virtual int n_input_dims() const = 0;
  virtual int n_output_dims() const = 0;
  virtual int n_neurons() const = 0;
  virtual int n_features_per_level() const = 0;
  virtual bool valid() const = 0;
  virtual size_t get_model_size() const = 0;
  virtual size_t get_mlp_size() const = 0;
  virtual size_t get_enc_size() const = 0;
  virtual size_t training_step() const = 0;
  virtual double training_loss() const = 0;
  virtual json serialize_params() const = 0;
  virtual void deserialize_params(const json& parameters) = 0;
  virtual json serialize_model() const = 0;
  virtual void deserialize_model(const json& config) = 0;
  virtual void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) = 0;
  virtual void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const = 0;
};

}
