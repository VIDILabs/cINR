//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "volumes.h"

#include "samplers/sampler.h"

#include "api.h"
#include "networks/nnapi.h"

#include <evaluation_kernel.h>

#include <cuda/cuda_buffer.h>
#include <cuda/cuda_math.h>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <vidi_progress_bar.h>
#include <vidi_highperformance_timer.h>
#include <vidi_filemap.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <ctime>

#define float_large 1e20f
#define float_small std::numeric_limits<float>::min() // to avoid division by zero
#define float_epsilon std::numeric_limits<float>::epsilon()
#define nearly_one 0.9999f

#ifdef ENABLE_LOGGING
#define logging() std::cout
#else
static std::ostream null_output_stream(0);
#define logging() null_output_stream
#endif

// make a private version of thrust::plus to avoid template instantiation conflicts ...
namespace {

// template<typename T>
// struct maximum_op {
//   typedef T first_argument_type;
//   typedef T second_argument_type;
//   typedef T result_type;
//   __host__ __device__ constexpr T operator()(const T& lhs, const T& rhs) const { return lhs < rhs ? rhs : lhs; }
// }; // end maximum

// template<typename T>
// struct minimum_op {
//   typedef T first_argument_type;
//   typedef T second_argument_type;
//   typedef T result_type;
//   __host__ __device__ constexpr T operator()(const T& lhs, const T& rhs) const { return lhs < rhs ? lhs : rhs; }
// }; // end minimum

// template<typename T>
// struct plus {
//   typedef T first_argument_type;
//   typedef T second_argument_type;
//   typedef T result_type;
//   __host__ __device__ constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }
// }; // end plus

// template<typename T>
// T parallel_sum_gpu(const T* __restrict__ data, size_t count, cudaStream_t stream = nullptr) {
//   const auto begin = thrust::device_ptr<const T>(data);
//   const auto end = begin + count;
//   return thrust::reduce(thrust::cuda::par.on(stream), begin, end, T(0), plus<T>());
// }

// template<typename T>
// void parallel_minmax_gpu(const T* __restrict__ data, size_t count, T& init_min, T& init_max, cudaStream_t stream = nullptr) {
//   const auto begin = thrust::device_ptr<const T>(data);
//   const auto end = begin + count;
//   init_min = thrust::reduce(thrust::cuda::par.on(stream), begin, end, init_min, minimum_op<T>());
//   init_max = thrust::reduce(thrust::cuda::par.on(stream), begin, end, init_max, maximum_op<T>());
// }

}

namespace vnr {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

__global__ void 
generate_coords(uint32_t n_elements, vec3i lower, vec3i size, vec3f rdims, float* __restrict__ coords)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elements)
    return;

  const uint64_t idx = 3 * i;

  const uint64_t stride = (uint64_t)size.x * size.y;
  const int32_t x = lower.x +  i % size.x;
  const int32_t y = lower.y + (i % stride) / size.x;
  const int32_t z = lower.z +  i / stride;

  coords[idx + 0] = (x+0.5f) * rdims.x;
  coords[idx + 1] = (y+0.5f) * rdims.y;
  coords[idx + 2] = (z+0.5f) * rdims.z;
}

template<int WIN_SIZE>
__global__ void 
compute_ssim(const uint32_t dimx, const uint32_t dimy, const uint32_t dimz,
             float* __restrict__ _fx, float* __restrict__ _fy, vec3i gdims,              
             float data_range, float cov_norm, float K1, float K2,
             float* __restrict__ out)
{
  const int32_t x = blockIdx.x * blockDim.x + threadIdx.x; if (x >= dimx) return;
  const int32_t y = blockIdx.y * blockDim.y + threadIdx.y; if (y >= dimy) return;
  const int32_t z = blockIdx.z * blockDim.z + threadIdx.z; if (z >= dimz) return;

  float ux = 0.f;
  float uy = 0.f;
  float uxx = 0.f;
  float uyy = 0.f;
  float uxy = 0.f;

  for (int kz = 0; kz < WIN_SIZE; ++kz) {
  for (int ky = 0; ky < WIN_SIZE; ++ky) {
  for (int kx = 0; kx < WIN_SIZE; ++kx) {

    const vec3i g = vec3i(x + kx, y + ky, z + kz);
    const uint32_t gidx = g.x + g.y * gdims.x + g.z * gdims.x * gdims.y;
    const float fx = _fx[gidx];
    const float fy = _fy[gidx];

    ux  += fx;
    uy  += fy;    
    uxx += fx * fx;
    uyy += fy * fy;
    uxy += fx * fy;

  }
  }
  }
    
  const float w = 1.f / (WIN_SIZE*WIN_SIZE*WIN_SIZE); // uniform filter
  ux  *= w;
  uy  *= w;
  uxx *= w;
  uyy *= w;
  uxy *= w;

  const float vx = cov_norm * (uxx - ux * ux);
  const float vy = cov_norm * (uyy - uy * uy);
  const float vxy = cov_norm * (uxy - ux * uy);

  const float R = data_range;
  const float C1 = (K1 * R) * (K1 * R);
  const float C2 = (K2 * R) * (K2 * R);

  const float A1 = 2 * ux * uy + C1;
  const float A2 = 2 * vxy + C2;
  const float B1 = ux * ux + uy * uy + C1;
  const float B2 = vx + vy + C2;
  const float D = B1 * B2;
  const float S = (A1 * A2) / D;

  out[x + y * dimx + z * dimx * dimy] = S;
}

void 
generate_grid_coords(float* d_coords, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream)
{
  util::linear_kernel(generate_coords, 0, stream, (uint32_t)grid_dims.long_product(), 
                      grid_origin, grid_dims, grid_spacing, (float*)d_coords);
}


// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

struct NeuralVolume::Impl
{
public:
  vec3i m_dims;
  affine3f m_transform;

  SimpleVolume* m_ref = nullptr; // optional
  vnrNetwork    m_net;

  MacroCell m_macrocell;

  struct {
    vec3i m_gdims{}; // number of grid points of the entire volume
    vec3i m_lower{};
    vec3i m_upper{};

    std::unique_ptr<GPUMatrixDynamic<float>> m_infer_x; // Auxiliary matrices for evaluation
    std::unique_ptr<GPUMatrixDynamic<float>> m_infer_y;
    std::unique_ptr<GPUColumnMatrix> m_train_x; // Auxiliary matrices for training
    std::unique_ptr<GPUColumnMatrix> m_train_y;
    std::unique_ptr<GPUColumnMatrix> m_test_x;
    std::unique_ptr<GPUColumnMatrix> m_test_y0;
    std::unique_ptr<GPUColumnMatrix> m_test_y1;
    GPUMemory<float> m_loss_buffer;
    
    // specific to fully decode mode
    cudaArray_t m_decoded_array{}; // TODO release memory correctly
    cudaTextureObject_t m_decoded_tex{};
    size_t m_num_slices_per_blob{16};
    GPUMemory<float> m_inference_buffer;

    cudaTextureObject_t& inference_tex() { return m_decoded_tex; }
  } m_trainer;

  TransferFunctionObject tfn;

  const size_t m_batch_size = 1 << 16;
  cudaStream_t m_infer_stream{};
  cudaStream_t m_train_stream{};

public:
  Impl(size_t batchsize) : m_batch_size(batchsize)
  {
    // CUDA_CHECK(cudaStreamCreate(&m_infer_stream));
    m_train_stream = m_infer_stream;
  }

  ~Impl()
  {
  }

  void resize_trainer(const vec3i lower, const vec3i upper, const vec3i gdims, const size_t batch_size)
  {
    assert(lower.x >= 0 && lower.x <= gdims.x && "valid volume index");
    assert(lower.y >= 0 && lower.y <= gdims.y && "valid volume index");
    assert(lower.z >= 0 && lower.z <= gdims.z && "valid volume index");
    assert(upper.x >= 0 && upper.x <= gdims.x && "valid volume index");
    assert(upper.y >= 0 && upper.y <= gdims.y && "valid volume index");
    assert(upper.z >= 0 && upper.z <= gdims.z && "valid volume index");
    
    const auto dims = upper - lower;
    const auto num_coords = (size_t)dims.x * dims.y * m_trainer.m_num_slices_per_blob;
    const auto num_coords_padded = util::next_multiple<size_t>(num_coords, 256);

    const auto INPUT_SIZE  = vnrGetCoordDims(m_net);
    const auto OUTPUT_SIZE = vnrGetValueDims(m_net);

    // create training & inference buffers
    m_trainer.m_inference_buffer = GPUMemory<float>(num_coords_padded * INPUT_SIZE);
    m_trainer.m_infer_x = std::make_unique<GPUColumnMatrix>(m_trainer.m_inference_buffer.data(), INPUT_SIZE, (uint32_t)num_coords_padded);
    m_trainer.m_infer_y = std::make_unique<GPUColumnMatrix>(OUTPUT_SIZE, (uint32_t)num_coords_padded);
    m_trainer.m_train_x = std::make_unique<GPUColumnMatrix>(INPUT_SIZE,  (uint32_t)batch_size);
    m_trainer.m_train_y = std::make_unique<GPUColumnMatrix>(OUTPUT_SIZE, (uint32_t)batch_size);
    m_trainer.m_test_x  = std::make_unique<GPUColumnMatrix>(m_trainer.m_train_x->data(), INPUT_SIZE,  (uint32_t)batch_size);
    m_trainer.m_test_y0 = std::make_unique<GPUColumnMatrix>(m_trainer.m_train_y->data(), OUTPUT_SIZE, (uint32_t)batch_size);
    m_trainer.m_test_y1 = std::make_unique<GPUColumnMatrix>(OUTPUT_SIZE, (uint32_t)batch_size);
    m_trainer.m_loss_buffer = GPUMemory<float>(batch_size);

    m_trainer.m_gdims = gdims;
    m_trainer.m_lower = lower;
    m_trainer.m_upper = upper;
  }

  void train(size_t steps, MacroCell* macrocell, bool verbose)
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; return;
    }

    auto stream = m_train_stream;

    const vec3f lower = vec3f(m_trainer.m_lower) / vec3f(m_trainer.m_gdims);
    const vec3f upper = vec3f(m_trainer.m_upper) / vec3f(m_trainer.m_gdims);

    ProgressBar bar("[train]");
    for (int i = 0; i < steps; ++i) {
      m_ref->sampler->take_samples(m_trainer.m_train_x->data(), m_trainer.m_train_y->data(), m_batch_size, stream, lower, upper);
      vnrTrain(m_net, (vnrDevicePtr)m_trainer.m_train_x->data(), (vnrDevicePtr)m_trainer.m_train_y->data(), m_batch_size, 1, stream);

      // update macrocell
      if (macrocell) {
        macrocell->update_explicit((vec3f*)m_trainer.m_train_x->data(), (float*)m_trainer.m_train_y->data(), m_batch_size, stream);
      }
      if (verbose) bar.update((float)i / steps);
    }
    if (verbose) bar.finalize();
  }

  void test(float* loss)
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; return;
    }

    auto stream = m_infer_stream;

    const vec3f lower = vec3f(m_trainer.m_lower) / vec3f(m_trainer.m_gdims);
    const vec3f upper = vec3f(m_trainer.m_upper) / vec3f(m_trainer.m_gdims);

    m_ref->sampler->take_samples(m_trainer.m_test_x->data(), m_trainer.m_test_y0->data(), m_batch_size, stream, lower, upper);
    vnrInfer(m_net, (vnrDevicePtr)m_trainer.m_test_x->data(), (vnrDevicePtr)m_trainer.m_test_y1->data(), m_batch_size, stream);

    util::parallel_for_gpu(0, stream, m_batch_size, 
    [
      target=(float*)m_trainer.m_test_y0->data(), 
      pred  =(float*)m_trainer.m_test_y1->data(), 
      output=m_trainer.m_loss_buffer.data()
    ] 
    __device__ (size_t i) {
      output[i] = l1_loss(pred[i], target[i]);
    });

    *loss = parallel_sum_gpu(m_trainer.m_loss_buffer.data(), m_batch_size, stream) / m_batch_size;
  }

  void progressively_decode()
  {
    static int b = 0; // blob index

    auto stream = m_infer_stream;

    if (!m_trainer.m_decoded_tex) {
      CreateArray3DScalar<float>(m_trainer.m_decoded_array, m_trainer.m_decoded_tex, m_dims, true);
    }

    const auto dims = m_trainer.m_upper - m_trainer.m_lower;
    const auto nz = std::min((int)m_trainer.m_num_slices_per_blob, dims.z - b * (int)m_trainer.m_num_slices_per_blob);

    util::linear_kernel(generate_coords, 0, stream, (uint32_t)((size_t)dims.x * dims.y * nz), 
                        vec3i(m_trainer.m_lower.x, m_trainer.m_lower.y, m_trainer.m_lower.z + b * (uint32_t)m_trainer.m_num_slices_per_blob), 
                        vec3i(dims.x, dims.y, nz), 
                        vec3f(1.f / m_trainer.m_gdims.x, 1.f / m_trainer.m_gdims.y, 1.f / m_trainer.m_gdims.z), 
                        m_trainer.m_inference_buffer.data());

    vnrInfer(m_net, (vnrDevicePtr)m_trainer.m_infer_x->data(), (vnrDevicePtr)m_trainer.m_infer_y->data(), 
             (uint32_t)(dims.x * dims.y * nz), stream);

    // copy result to inference array
    cudaMemcpy3DParms param = { 0 };
    param.srcPos = make_cudaPos(m_trainer.m_lower.x, m_trainer.m_lower.y, m_trainer.m_lower.z);
    param.dstPos = make_cudaPos(m_trainer.m_lower.x, m_trainer.m_lower.y, m_trainer.m_lower.z + b * m_trainer.m_num_slices_per_blob);
    param.srcPtr = make_cudaPitchedPtr(m_trainer.m_infer_y->data(), dims.x * sizeof(float), dims.x, dims.y); 
    param.dstArray = m_trainer.m_decoded_array;
    param.extent = make_cudaExtent(dims.x, dims.y, nz);
    param.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3DAsync(&param, stream));

    TRACE_CUDA;

    b++; if (b * m_trainer.m_num_slices_per_blob >= dims.z) b = 0;
  }

  void decode_volume(float* output, vec3i dims) const
  {
    const auto rdims = 1.f / dims;
    const auto batch = vec3i(dims.x, dims.y, 1);
    const auto count = util::next_multiple<size_t>(batch.long_product(), 256);
    GPUMemory<vec3f> slice_input(count);
    GPUMemory<float> slice_value(count);
    ProgressBar bar("[decoding volume]");
    uint64_t bw = 0;
    for (int z = 0; z < dims.z; ++z) {
      util::linear_kernel(generate_coords, 0, 0, (uint32_t)count, vec3i(0,0,z), batch, rdims, (float*)slice_input.data());
      GPUColumnMatrix x((float*)slice_input.data(), 3, (uint32_t)count);
      GPUColumnMatrix y((float*)slice_value.data(), 1, (uint32_t)count);

      vnrInfer(m_net, (vnrDevicePtr)x.data(), (vnrDevicePtr)y.data(), (uint32_t)count, 0);

      slice_value.copy_to_host(output, count); output += count;
      bar.update((float)z / dims.z);
    }
    bar.finalize();
  }

  void save_inference_volume(std::string filename, vec3i dims) const // save the actual decoded volume
  {
    const auto rdims = 1.f / dims;
    const auto batch = vec3i(dims.x, dims.y, 1);
    const auto count = util::next_multiple<size_t>(batch.long_product(), 256);

    GPUMemory<vec3f> slice_input(count);
    GPUMemory<float> slice_value(count);
    std::vector<float> values(count);

    float value_max = -float_large;
    float value_min = +float_large;

    ProgressBar bar("[saving volume]");

    vidi::FileMap w = vidi::filemap_write_create(filename, sizeof(float) * count * dims.z);
    uint64_t bw = 0;
    for (int z = 0; z < dims.z; ++z) {
      util::linear_kernel(generate_coords, 0, 0, (uint32_t)count, vec3i(0,0,z), batch, rdims, (float*)slice_input.data());
      GPUColumnMatrix x((float*)slice_input.data(), 3, (uint32_t)count);
      GPUColumnMatrix y((float*)slice_value.data(), 1, (uint32_t)count);

      vnrInfer(m_net, (vnrDevicePtr)x.data(), (vnrDevicePtr)y.data(), (uint32_t)count, 0);

      slice_value.copy_to_host(values);
      vidi::filemap_random_write_update(w, bw, values.data(), sizeof(float) * count);

      parallel_minmax_gpu(slice_value.data(), count, value_min, value_max);

      bar.update((float)z / dims.z);
    }
    bar.finalize();
    vidi::filemap_close(w);
    logging() << "[saving volume] saved the inference volume to: " << filename << std::endl;
    logging() << "[saving volume] min=" << value_min << " max=" << value_max << std::endl;
  }

  void save_reference_volume(std::string filename, vec3i dims) const
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; return;
    }

    const auto rdims = 1.f / dims;
    const auto batch = vec3i(dims.x, dims.y, 1);
    const auto count = util::next_multiple<size_t>(batch.long_product(), 256);

    GPUMemory<vec3f> slice_input(count);
    GPUMemory<float> slice_value(count);
    std::vector<float> values(count);

    float value_max = -float_large;
    float value_min = +float_large;

    ProgressBar bar("[saving volume]");

    std::ofstream ofile("reference.bin", std::ios::out | std::ios::binary);
    if (!ofile) throw std::runtime_error("Cannot open file: reference.bin");

    for (int z = 0; z < dims.z; ++z) {
      m_ref->sampler->take_samples_grid(slice_input.data(), slice_value.data(), vec3i(0,0,z), batch, rdims, nullptr);
      slice_value.copy_to_host(values);

      ofile.write((char *)values.data(), sizeof(float) * count);

      parallel_minmax_gpu(slice_value.data(), count, value_min, value_max);

      bar.update((float)z / dims.z);
    }
    bar.finalize();

    ofile.close();
    if(!ofile.good()) throw std::runtime_error("Error occurred at writing reference.bin");

    logging() << "[saving volume] saved the reference volume to: " << "reference.bin" << std::endl;
    logging() << "[saving volume] min=" << value_min << " max=" << value_max << std::endl;
  }

  float get_mse(vec3i dims, bool quiet) const
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; return -1.f;
    }

    const vec3f rdims = 1.f / dims;

    const vec3i batch = min(vec3i(4096,16,16),dims);
    const auto N = util::next_multiple<size_t>(batch.long_product(), 256);

    GPUMemory<vec3f> coords(N);
    GPUMemory<float> values_inference(N);
    GPUMemory<float> values_reference(N);
    GPUColumnMatrix network_i((float*)coords.data(),   3, (uint32_t)N);
    GPUColumnMatrix network_o(values_inference.data(), 1, (uint32_t)N);

    float* loss = (float*)coords.data();

    // compute MSE
    double error_sum = 0.0;

    ProgressBar bar("[PSNR]");

    for (int z = 0; z < dims.z; z += batch.z) {
    for (int y = 0; y < dims.y; y += batch.y) {
    for (int x = 0; x < dims.x; x += batch.x) {

      const auto offset = vec3i(x, y, z);
      const auto block  = min(batch, dims - offset);
      const auto count = block.long_product();
      if (count == 0) continue;

      // reference
      m_ref->sampler->take_samples_grid(coords.data(), values_reference.data(), offset, block, rdims, nullptr);
      // inference
      vnrInfer(m_net, (vnrDevicePtr)network_i.data(), (vnrDevicePtr)network_o.data(), (uint32_t)count, 0);

      // squared error
      util::parallel_for_gpu(0, 0, count, 
      [pred=values_inference.data(), targ=values_reference.data(), out=loss] __device__ (size_t i) {
        out[i] = l2_loss((float)pred[i], (float)targ[i]);
      });

      // compute total error
      error_sum += parallel_sum_gpu(loss, count);
    }
    if (!quiet) bar.update((float)(z * dims.y + y) / (dims.y*dims.z));
    }
    if (!quiet) bar.update((float)z / dims.z);
    }
    if (!quiet) bar.finalize();

    return (float)(error_sum / dims.long_product());
  }

  float get_psnr(vec3i dims, bool quiet) const
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; return -1.f;
    }

    // convert MSE to PSNR
    const float range = 1.0; // NOTE we fix data range == 1 to match python implementation
    const float mse = get_mse(dims, quiet);
    return (float)(10. * log10(range * range / mse));
  }

  float get_mssim(vec3i dims, bool quiet) const
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; return -1.f;
    }

    const vec3f rdims = 1.f / dims;

    constexpr float K1 = 0.01f;
    constexpr float K2 = 0.03f;
    constexpr bool use_sample_covariance = true;

    constexpr int ndim = 3; // inputs are volumetric data
    constexpr int win_size = 7; // backwards compatibility
    constexpr int crop = win_size >> 1;
    constexpr int NP = win_size * win_size * win_size;
    // filter has already normalized by NP
    constexpr float cov_norm = use_sample_covariance 
      ? (float)NP / (NP - 1) // sample covariance
      : 1.f; // population covariance to match Wang et. al. 2004

    const vec3i batch = min(vec3i(4096,16,16),dims);
    const auto batch_count = util::next_multiple<size_t>(batch.long_product(), 256);

    const vec3i batch_grid = batch + win_size - 1;
    const auto batch_grid_count = util::next_multiple<size_t>(batch_grid.long_product(), 256);

    GPUMemory<float>   grid_input(3 * batch_grid_count);
    GPUMemory<float>   grid_inference(batch_grid_count);
    GPUMemory<float>   grid_reference(batch_grid_count);
    GPUColumnMatrix network_in (grid_input.data(),     3, (uint32_t)batch_grid_count);
    GPUColumnMatrix network_out(grid_inference.data(), 1, (uint32_t)batch_grid_count);
    float* output = grid_input.data();

    // probably should calculate the true value range
    const float data_range = 1.f;

    ProgressBar bar("[SSIM]");

    float ssim_sum = 0.f;
    for (int z = crop; z < dims.z - crop; z += batch.z) {
    for (int y = crop; y < dims.y - crop; y += batch.y) {
    for (int x = crop; x < dims.x - crop; x += batch.x) {
      const vec3i block_offset = vec3i(x,y,z);
      const vec3i block = min(batch, dims - crop - block_offset);
      const auto block_count = block.long_product();
      if (block_count == 0) continue;

      // compute grid values
      const vec3i block_grid_offset = block_offset - crop;
      const vec3i block_grid = block + win_size - 1;

      // reference & inference
      m_ref->sampler->take_samples_grid(grid_input.data(), grid_reference.data(), block_grid_offset, block_grid, rdims, 0);
      vnrInfer(m_net, (vnrDevicePtr)network_in.data(), (vnrDevicePtr)network_out.data(), (uint32_t)batch_grid_count, 0);

      // calculate SSIM
      util::trilinear_kernel(compute_ssim<win_size>, 0, 0, block.x, block.y, block.z, 
                       grid_reference.data(), grid_inference.data(), block_grid,
                       data_range, cov_norm, K1, K2, output);
      std::vector<float> S(block.long_product());
      grid_input.copy_to_host(S.data(), block.long_product());

      // compute total ssim
      ssim_sum += parallel_sum_gpu(output, block_count);
    }
    if (!quiet) bar.update((float)(z * dims.y + y) / (dims.y*dims.z));
    }
    if (!quiet) bar.update((float)z / dims.z);
    }
    if (!quiet) bar.finalize();

    return ssim_sum / (dims - win_size + 1).long_product();
  }

  void set_network(vec3i _dims, json config, SimpleVolume* reference, bool use_reference_macrocell)
  {
    if (use_reference_macrocell && !(reference && reference->texture())) {
      std::cerr << GDT_TERMINAL_RED 
                << "[vnr] ground truth macrocell unavailable with training mode: " 
                << reference->mode 
                << GDT_TERMINAL_RESET << std::endl;
    }
    m_ref = reference;
    use_reference_macrocell = m_ref && m_ref->texture() && use_reference_macrocell;

    // sync the shape between groundtruth and neural volume
    if (m_ref) {
      m_dims = m_ref->get_data_dims();
      m_transform = m_ref->get_data_transform();
    }
    else {
      m_dims = _dims; // '_dims' should not be used
      m_transform = affine3f::translate(vec3f(m_dims) / vec3f(-2.f)) * affine3f::scale(vec3f(m_dims));
    }

    m_net = vnrCreateNetwork(config);

    resize_trainer(vec3i(0), m_dims, m_dims, m_batch_size);

    // construct macrocell

    // it DOES make sense to render neural representation WITHOUT a texture
    if (use_reference_macrocell) {
      m_macrocell.set_external(reference->macrocell);
    }
    else {
      m_macrocell.set_shape(m_dims);
      m_macrocell.allocate();
      // std::cout << GDT_TERMINAL_RED << "Hacking!!! let reference to use trained macrocell" << GDT_TERMINAL_RESET << std::endl;
      // reference->macrocell.set_external(m_macrocell);
    }

    CUDA_SYNC_CHECK();
  }

  void update_inference_macrocell(const DeviceTransferFunction& tfn) 
  {
    m_macrocell.update_max_opacity(tfn, m_infer_stream);
  }

  vec2f get_macrocell_psnr() const
  {
    if (!m_ref) {
      std::cerr << "[error]: missing a reference volume." << std::endl; 
      return -1.f;
    }

    if (!m_ref->macrocell.allocated()) {
      std::cerr << "[error]: the reference volume doesnot contain a macrocell." << std::endl;
      return -1.f;
    }

    if (m_macrocell.is_external()) {
      std::cerr << "[error] cannot compute PSNR for macrocell." << std::endl;
      return -1.f;
    }

    const auto compute_psnr = [] (float* pred, float* targ, float *loss, size_t count) -> float {
      util::parallel_for_gpu(0, 0, count, 
      [pred=pred,targ=targ,loss=loss] __device__ (size_t i) {
        loss[i] = l2_loss((float)pred[i], (float)targ[i]);
      });

      // compute MSE
      const float mse = parallel_sum_gpu(loss, count) / count;

      // compute psnr
      const float range = 1.0f;
      return (float)(10. * log10(range * range / mse));
    };

    const size_t n_mcs = m_macrocell.dims().long_product();
    GPUMemory<float> loss(n_mcs);
    GPUMemory<float> pred(n_mcs);
    GPUMemory<float> targ(n_mcs);

    util::parallel_for_gpu(0, 0, n_mcs, 
    [
      mc_pred=m_macrocell.d_value_range(), 
      mc_targ=m_ref->macrocell.d_value_range(), 
      out_pred=pred.data(),
      out_targ=targ.data()
    ] __device__ (size_t i) {
      out_pred[i] = mc_pred[i].x + 1;
      out_targ[i] = mc_targ[i].x + 1;
    });

    float min_psnr = compute_psnr(pred.data(), targ.data(), loss.data(), n_mcs);

    util::parallel_for_gpu(0, 0, n_mcs, 
    [
      mc_pred=m_macrocell.d_value_range(), 
      mc_targ=m_ref->macrocell.d_value_range(), 
      out_pred=pred.data(),
      out_targ=targ.data()
    ] __device__ (size_t i) {
      out_pred[i] = mc_pred[i].y - 1;
      out_targ[i] = mc_targ[i].y - 1;
    });

    float max_psnr = compute_psnr(pred.data(), targ.data(), loss.data(), n_mcs);

    return vec2f(min_psnr, max_psnr);
  }
};


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

NeuralVolume::NeuralVolume(size_t batchsize) : pimpl(new Impl(batchsize)) {}

NeuralVolume::~NeuralVolume() { 
  pimpl.reset(); 
  free_temporary_gpu_memory_by_tcnn();
}

void 
NeuralVolume::set_network_from_json(vec3i dims, const json& config, SimpleVolume* reference, bool use_reference_macrocell)
{
  pimpl->set_network(dims, config, reference, use_reference_macrocell);
}

void
NeuralVolume::set_network(vec3i dims, std::string filename, SimpleVolume* reference, bool use_reference_macrocell)
{
  std::ifstream file(filename);
  pimpl->set_network(dims, json::parse(file, nullptr, true, true), reference, use_reference_macrocell);
}

void 
NeuralVolume::set_network(std::string filename)
{
  std::ifstream file(filename);
  pimpl->m_net = vnrCreateNetwork(json::parse(file, nullptr, true, true));
}

void 
NeuralVolume::set_network_from_json(const json& config)
{
  pimpl->m_net = vnrCreateNetwork(config);
}

void
NeuralVolume::set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r)
{
  pimpl->tfn.set_transfer_function(c, o, r, pimpl->m_infer_stream);

  // Update reference macrocell data once per transfer function update because the macrocell value
  // ranges are pre-computed. The inference macrocell data is updated for every training step because
  // the inference value ranges are online-trained.
  pimpl->update_inference_macrocell(pimpl->tfn.tfn);

  TRACE_CUDA;
}

void
NeuralVolume::statistics(Statistics& stat)
{
  stat.step = vnrGetStep(pimpl->m_net);
  stat.loss = vnrGetLoss(pimpl->m_net);
}

void
NeuralVolume::train(size_t steps, bool fast_mode, bool verbose)
{
  if (!pimpl->m_net) return;

  auto* mcdata = (fast_mode && pimpl->m_macrocell.is_external()) ? nullptr : &pimpl->m_macrocell;

  pimpl->train(steps, mcdata, verbose);

  if (!fast_mode) pimpl->update_inference_macrocell(pimpl->tfn.tfn);
}

void 
NeuralVolume::test(float* loss)
{
  if (!pimpl->m_net) return;
  pimpl->test(loss);
}

void
NeuralVolume::decode_progressive()
{
  if (!pimpl->m_net) return;
  pimpl->progressively_decode();
}

float 
NeuralVolume::get_mse(vec3i resolution, bool quiet) const
{
  return pimpl->get_mse(resolution, quiet);
}

float 
NeuralVolume::get_psnr(vec3i resolution, bool quiet) const
{
  return pimpl->get_psnr(resolution, quiet);
}

float 
NeuralVolume::get_mssim(vec3i resolution, bool quiet) const
{
  return pimpl->get_mssim(resolution, quiet);
}

vec2f 
NeuralVolume::get_macrocell_psnr() const
{
  return pimpl->get_macrocell_psnr();
}

void 
NeuralVolume::decode_volume(float* output, vec3i resolution) const
{
  pimpl->decode_volume(output, resolution);
}

void
NeuralVolume::save_inference_volume(std::string filename, vec3i resolution) const
{
  pimpl->save_inference_volume(filename, resolution);
}

void
NeuralVolume::save_reference_volume(std::string filename, vec3i resolution) const
{
  pimpl->save_reference_volume(filename, resolution);
}

void 
NeuralVolume::save_params_to_json(json& root) const
{
  const auto& mc = pimpl->m_macrocell;
  root["volume"] = {
    { "dims", {
      { "x", pimpl->m_dims.x },
      { "y", pimpl->m_dims.y },
      { "z", pimpl->m_dims.z }
    }}
  };
  root["macrocell"] = {
    { "groundtruth", pimpl->m_macrocell.is_external() },
    { "dims", {
      { "x", mc.dims().x },
      { "y", mc.dims().y },
      { "z", mc.dims().z },
    }},
    { "spacings", {
      { "x", mc.spacings().x },
      { "y", mc.spacings().y },
      { "z", mc.spacings().z },
    }},
    { "data", gpu_memory_to_json_binary(mc.d_value_range(), mc.dims().long_product() * sizeof(vec2f)) },
  };
  vnrExportConfig(pimpl->m_net, root["model"]);
  vnrExportParams(pimpl->m_net, root["parameters"]);
}

void
NeuralVolume::save_params(std::string filename) const
{
  const auto& mc = pimpl->m_macrocell;

  logging() << "saving parameters to " << filename << std::endl;

  json root; save_params_to_json(root);

  const auto broot = json::to_bson(root);
  std::ofstream ofs(filename, std::ios::binary | std::ios::out);
  ofs.write((char*)broot.data(), broot.size());
  ofs.close();

  logging() << "  macrocell data ... " << util::prettyBytes(mc.dims().long_product() * sizeof(vec2f)) << std::endl;
  logging() << "  network data ..... " << util::prettyBytes(vnrGetModelSizeInBytes(pimpl->m_net)) << std::endl;
  logging() << "  total data ....... " << util::prettyBytes(broot.size()) << std::endl;
  logging() << "total time";
}

void 
NeuralVolume::load_params_from_json(const json& root)
{
    const auto& mc = pimpl->m_macrocell;

    vidi::StackTimer time(logging());

    if (root.contains("volume")) {
      const vec3i dims = vec3i(root["volume"]["dims"]["x"].get<int>(),
                               root["volume"]["dims"]["y"].get<int>(),
                               root["volume"]["dims"]["z"].get<int>());
      if (dims != pimpl->m_dims) {
        throw std::runtime_error("mismatch data dimension");
      }
    }

    if (root.contains("macrocell")) {
      const vec3i mcdims = vec3i(root["macrocell"]["dims"]["x"].get<int>(),
                                 root["macrocell"]["dims"]["y"].get<int>(),
                                 root["macrocell"]["dims"]["z"].get<int>());
      const vec3f mcspac = vec3f(root["macrocell"]["spacings"]["x"].get<float>(),
                                 root["macrocell"]["spacings"]["y"].get<float>(),
                                 root["macrocell"]["spacings"]["z"].get<float>());

      // if (mcdims != pimpl->m_macrocell_dims)     throw std::runtime_error("mismatch macrocell dimension");
      // if (mcspac != pimpl->m_macrocell_spacings) throw std::runtime_error("mismatch macrocell spacings" );

      if (mcdims != pimpl->m_macrocell.dims() || mcspac != pimpl->m_macrocell.spacings()) {
        printf("mismatch macrocell dimension or spacing");

        pimpl->m_macrocell.set_dims(mcdims);
        pimpl->m_macrocell.set_spacings(mcspac);
        pimpl->m_macrocell.allocate();

        // std::cout << GDT_TERMINAL_RED << "Hacking!!! let reference to use trained macrocell" << GDT_TERMINAL_RESET << std::endl;
        // pimpl->m_ref->macrocell.set_external(pimpl->m_macrocell);
      }

      json_binary_to_gpu_memory(root["macrocell"]["data"], mc.d_value_range(), mc.dims().long_product() * sizeof(vec2f));
      pimpl->update_inference_macrocell(pimpl->tfn.tfn);
      if (root["macrocell"].contains("groundtruth")) {
        logging() << "[network] use GT macrocell = " << root["macrocell"]["groundtruth"].get<bool>() << std::endl;
      }
      logging() << "[network] macrocell dims = " << mcdims << std::endl;
    }

    pimpl->m_net = vnrCreateNetwork(root);

    logging() << "[network] time = ";
}

void
NeuralVolume::load_params(std::string filename)
{
  logging() << "[network] loading parameters from: " << filename << std::endl;

  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (file.read(buffer.data(), size)) {
    const auto root = json::from_bson(buffer);
    load_params_from_json(root);
  }
}

const cudaTextureObject_t&
NeuralVolume::texture() const
{
  return pimpl->m_trainer.inference_tex();
}

ValueType
NeuralVolume::get_data_type() const
{
  return ValueType::VALUE_TYPE_FLOAT;
}

uint32_t
NeuralVolume::get_num_blobs() const
{
  const auto dims = pimpl->m_dims;
  const auto num_slices_per_blob = (uint32_t)pimpl->m_trainer.m_num_slices_per_blob;
  return (dims.z + num_slices_per_blob - 1) / num_slices_per_blob;
}

uint32_t  
NeuralVolume::get_mlp_size() const
{
  return vnrGetModelNetworkSizeInBytes(pimpl->m_net);
}

uint32_t  
NeuralVolume::get_enc_size() const
{
  return vnrGetModelEncoderSizeInBytes(pimpl->m_net);
}

// range1f
// NeuralVolume::get_data_value_range() const
// {
//   return range1f(0, 1);
// }

vec3i
NeuralVolume::get_data_dims() const
{
  return (vec3i)pimpl->m_dims;
}

vec2f* 
NeuralVolume::get_macrocell_value_range() const
{
  return pimpl->m_macrocell.d_value_range();
}

float*
NeuralVolume::get_macrocell_max_opacity() const
{
  return pimpl->m_macrocell.d_max_opacity();
}

vec3i 
NeuralVolume::get_macrocell_dims() const
{
  return pimpl->m_macrocell.dims();
}

vec3f 
NeuralVolume::get_macrocell_spacings() const
{
  return pimpl->m_macrocell.spacings();
}

affine3f 
NeuralVolume::get_data_transform() const
{
  return pimpl->m_transform;
}

void 
NeuralVolume::set_data_transform(affine3f transform)
{
  pimpl->m_transform = transform;
}

// void* 
// NeuralVolume::get_network() const
// {
//   return pimpl->m_neural->network_direct_access();
// }

int 
NeuralVolume::get_network_width() const
{
  return vnrGetNumNeurons(pimpl->m_net);
}

int 
NeuralVolume::get_network_features_per_level() const
{
  return vnrGetNumFeaturesPerLevel(pimpl->m_net);
}

void 
NeuralVolume::inference(int len, const float* d_input, float* d_output, cudaStream_t stream)
{
  vnrInfer(pimpl->m_net, (vnrDevicePtr)d_input, (vnrDevicePtr)d_output, len, stream);
}

size_t NeuralVolume::max_nbytes_allocated_by_tcnn()
{
  static std::atomic<size_t> maximum_value{0}; 

  size_t value = tot_nbytes_allocated_by_tcnn();
  size_t prev_value = maximum_value;
  while(prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value)) {}

  return maximum_value;
}

size_t NeuralVolume::tot_nbytes_allocated_by_tcnn()
{
  return TCNN_NAMESPACE :: total_n_bytes_allocated();
}

void NeuralVolume::free_temporary_gpu_memory_by_tcnn()
{
  max_nbytes_allocated_by_tcnn(); // foce an update
  TCNN_NAMESPACE :: free_all_gpu_memory_arenas();
  // std::cout << "[vnr] free temporary gpu memory" << std::endl;
}

} // namespace vnr
