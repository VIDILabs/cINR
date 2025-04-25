//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "tcnn_network.h"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/gpu_memory_json.h>

#define TCNN_NEW_API // TODO add automatic check for new API

#ifdef ENABLE_LOGGING
#define logging() std::cout
#else
static std::ostream null_output_stream(0);
#define logging() null_output_stream
#endif

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

using precision_t = TCNN_NAMESPACE :: network_precision_t;

using Loss                     = TCNN_NAMESPACE :: Loss<precision_t>;
using Optimizer                = TCNN_NAMESPACE :: Optimizer<precision_t>;
using Trainer                  = TCNN_NAMESPACE :: Trainer<float, precision_t, precision_t>;
using NetworkWithInputEncoding = TCNN_NAMESPACE :: NetworkWithInputEncoding<precision_t>;
using Encoding  = TCNN_NAMESPACE :: Encoding<precision_t>;

using TCNN_NAMESPACE :: create_loss;
using TCNN_NAMESPACE :: create_optimizer;
using TCNN_NAMESPACE :: create_grid_encoding;

using ForwardContext = Trainer :: ForwardContext;

using network_t = std::shared_ptr<NetworkWithInputEncoding>;

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void 
tcnn_inference(network_t handler, cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output) {
  try {
    handler->inference(stream, input, output);
  }
  catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return;
  }
}

}

using namespace tcnn_impl;

struct TcnnNetwork::Impl {
  mutable std::shared_ptr<Loss> loss;
  mutable std::shared_ptr<NetworkWithInputEncoding> network;
  mutable std::shared_ptr<Trainer> trainer;
  std::unique_ptr<ForwardContext> ctx;
  json model_opts;
  uint64_t training_step = 0;
};

TcnnNetwork::~TcnnNetwork() {}

TcnnNetwork::TcnnNetwork(int n_input_dims, int n_output_dims) 
  : m(new Impl()), INPUT_SIZE(n_input_dims), OUTPUT_SIZE(n_output_dims)
{
}

bool TcnnNetwork::valid() const { return m->trainer.get() != nullptr; }

size_t TcnnNetwork::get_model_size() const { return sizeof(precision_t) * m->network->n_params(); }

size_t TcnnNetwork::get_mlp_size() const { return sizeof(precision_t) * m->network->m_network->n_params(); }

size_t TcnnNetwork::get_enc_size() const { return sizeof(precision_t) * m->network->m_encoding->n_params(); }

size_t TcnnNetwork::training_step() const { return m->training_step; }

double TcnnNetwork::training_loss() const { return m->trainer->loss(0, *m->ctx); }

json TcnnNetwork::serialize_params() const { return m->trainer->serialize(); }

void TcnnNetwork::deserialize_params(const json& parameters) { m->trainer->deserialize(parameters); }

json TcnnNetwork::serialize_model() const { return m->model_opts; }

void TcnnNetwork::deserialize_model(const json& config) {
  TRACE_CUDA;

  json loss_opts = config.value("loss", json::object());
  json encoding_opts = config.value("encoding", json::object());
  json network_opts = config.value("network", json::object());
  json optimizer_opts = config.value("optimizer", json::object());

  m->model_opts["loss"] = loss_opts;
  m->model_opts["encoding"] = encoding_opts;
  m->model_opts["network"] = network_opts;

  if (network_opts["otype"] == "FullyFusedMLP") {
    N_NEURONS = network_opts["n_neurons"].get<int>();
    logging() << "[network] WIDTH = " << N_NEURONS << std::endl;
  }
  else {
    N_NEURONS = -1;
    logging() << "[network] other MLP format" << std::endl;
  }

  if (encoding_opts["otype"] == "HashGrid") {
    N_FEATURES_PER_LEVEL = encoding_opts["n_features_per_level"].get<int>();
    logging() << "[network] N_FEATURES_PER_LEVEL = " << N_FEATURES_PER_LEVEL << std::endl;
  }
  else {
    N_FEATURES_PER_LEVEL = -1;
    logging() << "[network] other encoding method" << std::endl;
  }

  m->loss.reset();
  m->network.reset();
  m->trainer.reset();

  TRACE_CUDA;

  try {
    auto optimizer = std::shared_ptr<Optimizer>{ create_optimizer<precision_t>(optimizer_opts) };
    // NOTE: It is important to manually create the grid encoding here, despite the more convienient constructor exists.
    //       Not doing so will lead to the following error with linking against the pytorch extension: 
    //             CUDA sync error (.../core/networks/tcnn_network.h: line 202): __global__ function call is not configured
    //       The reason is unclear.
    auto encoder = std::shared_ptr<Encoding>{ create_grid_encoding<precision_t>(INPUT_SIZE, encoding_opts) };
    m->loss = std::shared_ptr<Loss>{ create_loss<precision_t>(loss_opts) };
    m->network = std::make_shared<NetworkWithInputEncoding>(encoder, OUTPUT_SIZE, network_opts);
    m->trainer = std::make_shared<Trainer>(m->network, optimizer, m->loss, (uint32_t)time(NULL));
  }
  catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }

  m->training_step = 0;
  logging() << "[network] total # of parameters = " << m->network->n_params() << std::endl;

  TRACE_CUDA;
}

void TcnnNetwork::train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) {
  TRACE_CUDA;
  try {
    m->ctx = m->trainer->training_step(stream, input, target);
    ++m->training_step;
  }
  catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    m->loss.reset();
    m->network.reset();
    m->trainer.reset();
    return;
  }
  TRACE_CUDA;
}

void TcnnNetwork::infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const {
  TRACE_CUDA;
  try {
    m->network->inference(stream, input, output);
  }
  catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    m->loss.reset();
    m->network.reset();
    m->trainer.reset();
    return;
  }
  TRACE_CUDA;
}

} // namespace vnr
