//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "api_internal.h"

#include "core/networks/tcnn_network.h"

#include <cuda.h>

#include <type_traits>

#include <mutex>

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

static_assert(std::is_same<vnrDevicePtr, CUdeviceptr>::value, 
  "vnrDevicePtr must be a valid CUDA device pointer type");

namespace vnr {

struct NetworkContext
{
  std::mutex mtx;
  std::unique_ptr<AbstractNetwork> net;
};

}

#define VNR_INPUT_DIMS  3
#define VNR_OUTPUT_DIMS 1

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

using namespace vnr;

vnrNetwork vnrCreateNetwork(const vnrJson& input) {
  auto ret = std::make_shared<NetworkContext>();

  vnrJson config;

  // check if input is a filename
  if (input.is_string()) {
    std::ifstream file(input.get<std::string>());
    config = vnrJson::parse(file, nullptr, true, true);
  }
  else {
    config = input;
  }

  ret->net = std::make_unique<TcnnNetwork>(VNR_INPUT_DIMS, VNR_OUTPUT_DIMS);
  
  // check if input is a network configuration or a parameter
  if (config.contains("model")) {
    std::cout << std::endl << "[network] reset model as: " << config["model"].dump(2) << std::endl;
    ret->net->deserialize_model(config["model"]);
    // load parameters if available
    if (config.contains("parameters")) {
      ret->net->deserialize_params(config["parameters"]);
    }
    else { // this is the old format
      ret->net->deserialize_params(config);
    }
  }
  else {
    ret->net->deserialize_model(config);
  }

  std::cout << "[network] size = " << util::prettyBytes(ret->net->get_model_size()) << std::endl;

  return ret;
}

void vnrTrain(vnrNetwork self, vnrDevicePtr d_coords, vnrDevicePtr d_values, size_t batchsize, size_t steps, vnrDeviceStream stream) {
  std::lock_guard<std::mutex> guard(self->mtx);
  const auto INPUT_SIZE  = self->net->n_input_dims ();
  const auto OUTPUT_SIZE = self->net->n_output_dims();
  GPUColumnMatrix coords((float*)d_coords, INPUT_SIZE,  (uint32_t)batchsize);
  GPUColumnMatrix values((float*)d_values, OUTPUT_SIZE, (uint32_t)batchsize);
  for (int i = 0; i < steps; ++i) {
    self->net->train(coords, values, (cudaStream_t)stream);
  }
}

void vnrInfer(vnrNetwork self, vnrDevicePtr d_coords, vnrDevicePtr d_values, size_t batchsize, vnrDeviceStream stream) {
  std::lock_guard<std::mutex> guard(self->mtx);
  const auto INPUT_SIZE  = self->net->n_input_dims ();
  const auto OUTPUT_SIZE = self->net->n_output_dims();
  GPUColumnMatrix coords((float*)d_coords, INPUT_SIZE,  (uint32_t)batchsize);
  GPUColumnMatrix values((float*)d_values, OUTPUT_SIZE, (uint32_t)batchsize);
  self->net->infer(coords, values, (cudaStream_t)stream);
}

double vnrGetLoss(vnrNetwork self) {
  return self->net->training_loss();
}

size_t vnrGetStep(vnrNetwork self) {
  return self->net->training_step();
}

uint32_t vnrGetModelSizeInBytes(vnrNetwork self) {
  return self->net->get_model_size();
}

uint32_t vnrGetModelNetworkSizeInBytes(vnrNetwork self) {
  return self->net->get_mlp_size();
}

uint32_t vnrGetModelEncoderSizeInBytes(vnrNetwork self) {
  return self->net->get_enc_size();
}

void vnrExportConfig(vnrNetwork self, vnrJson& config) {
  config = self->net->serialize_model();
}

void vnrExportParams(vnrNetwork self, vnrJson& params) {
  params = self->net->serialize_params();
}

uint32_t vnrGetCoordDims(vnrNetwork self) {
  return self->net->n_input_dims();
}

uint32_t vnrGetValueDims(vnrNetwork self) {
  return self->net->n_output_dims();
}

uint32_t vnrGetNumNeurons(vnrNetwork self) {
  return self->net->n_neurons();
}

uint32_t vnrGetNumFeaturesPerLevel(vnrNetwork self) { 
  return self->net->n_features_per_level();
}
