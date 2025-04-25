#include "space_partition.h"

#include "math_def.h"

#include "../../api.h"

#include <fstream>

using namespace ovr::misc;

namespace ovr {
namespace nncache {

static __global__ void MAYBE_UNUSED
value_range_kernel(const vec3i volumeDims, cudaTextureObject_t volumeTexture,
                   const vec3i mcDims, const uint32_t mcWidth, 
                   /* OUTPUT */ range1f* __restrict__ mcData)
{
  const uint32_t mcDimsX = mcDims.x;
  const uint32_t mcDimsY = mcDims.y;
  const uint32_t mcDimsZ = mcDims.z;

  // 3D kernel launch
  vec3i mcID(threadIdx.x + blockIdx.x * blockDim.x, 
             threadIdx.y + blockIdx.y * blockDim.y,
             threadIdx.z + blockIdx.z * blockDim.z);

  if (mcID.x >= mcDimsX) return;
  if (mcID.y >= mcDimsY) return;
  if (mcID.z >= mcDimsZ) return;

  const int mcIdx = mcID.x + mcDimsX * (mcID.y + mcDimsY * mcID.z);
  range1f& mc = mcData[mcIdx];

  if (volumeTexture == 0) {
    mc.lo = 0.f - 1.f;
    mc.hi = 1.f + 1.f;
    return;
  }

  // compute begin/end of VOXELS for this macro-cell
  const vec3i begin = max(mcID  * vec3i(mcWidth) - 1, vec3i(0));
  const vec3i end   = min(begin + vec3i(mcWidth) + /* plus one for tri-lerp!*/ 1, volumeDims);

  range1f valueRange;
  for (int iz = begin.z; iz < end.z; iz++)
    for (int iy = begin.y; iy < end.y; iy++)
      for (int ix = begin.x; ix < end.x; ix++) {
        float f;
        tex3D(&f, volumeTexture, (ix + 0.5f) / volumeDims.x, (iy + 0.5f) / volumeDims.y, (iz + 0.5f) / volumeDims.z);
        valueRange.extend(f);
      }

  // NOTE: we modify the range here to make it compatible with other implementations
  mc.lo = valueRange.lo - 1.f;
  mc.hi = valueRange.hi + 1.f;
}

void set_space_partition(SingleSpacePartiton& sp, std::string fname, range1f range, cudaStream_t stream)
{
  TRACE_CUDA;

  vnrJson root;
  vnrLoadJsonBinary(root, fname);

  if (root.contains("macrocell")) {
      const nlohmann::json::binary_t data = root["macrocell"]["data"];

      // ----
      // size_t numVec2f = data.size() / sizeof(vec2f);
      // const vec2f* vecData = reinterpret_cast<const vec2f*>(data.data());

      // std::string filename = "Macrocell_data.json";
      // std::ofstream outFile(filename);

      // outFile << root["macrocell"]["dims"]["x"].get<int>() << ' ';
      // outFile << root["macrocell"]["dims"]["y"].get<int>() << ' ';
      // outFile << root["macrocell"]["dims"]["z"].get<int>() << "\n";
      // outFile << root["macrocell"]["spacings"]["x"].get<float>() << ' ';
      // outFile << root["macrocell"]["spacings"]["y"].get<float>() << ' ';
      // outFile << root["macrocell"]["spacings"]["z"].get<float>() << "\n";
      // for (size_t i = 0; i < numVec2f; ++i) {
      //     outFile << "vec2f[" << i << "] = (" << vecData[i].x << ", " << vecData[i].y << ") | Range: " << std::fabs((vecData[i].x+1) - (vecData[i].y-1)) << "\n";
      // }

      // outFile.close();
      // std::cout << "Data saved to " << filename << std::endl;

      // ----

      CUDA_CHECK(cudaMemcpyAsync(sp.impl.d_value_range(), data.data(), sp.impl.dims().long_product() * sizeof(vec2f), cudaMemcpyHostToDevice, stream));
  }
  else 
    throw std::runtime_error("Failed to open Macrocell data in " + fname);

  // auto data = CreateArray3DScalarFromFile(fname.c_str(), sp.impl.dims(), ovr::VALUE_TYPE_FLOAT2, 0, false);
  // CUDA_CHECK(cudaMemcpyAsync(sp.impl.d_value_range(), data->data(), sp.impl.dims().long_product() * sizeof(vnr::vec2f), cudaMemcpyHostToDevice, stream));
  // data.reset(); // delete the data from memory
  
  TRACE_CUDA;

  if (range.empty()) {
    util::parallel_for_gpu(stream, sp.device.dims.long_product(), [sp, range] 
      __device__ (size_t i) {
        sp.device.value_ranges[i].lo = sp.device.value_ranges[i].lo ;
        sp.device.value_ranges[i].hi = sp.device.value_ranges[i].hi ;
      }
    );
  }
  else {
    util::parallel_for_gpu(stream, sp.device.dims.long_product(), [sp, range] 
      __device__ (size_t i) {
        sp.device.value_ranges[i].lo = (sp.device.value_ranges[i].lo - range.lo) / (range.hi - range.lo) ;
        sp.device.value_ranges[i].hi = (sp.device.value_ranges[i].hi - range.lo) / (range.hi - range.lo) ;
      }
    );
  }

  TRACE_CUDA;
}

} // namespace nncache
} // namespace ovr

//---------------------------------------------------------------------------------------------------
// 
//---------------------------------------------------------------------------------------------------

namespace tdns
{

template<typename T>
static __global__ void 
macrocell_range(const T* input, 
                const vnr::vec3i vol_dims,
                const vnr::vec3i mc_dims, 
                const vnr::vec3i mc_size, 
                vnr::range1f* __restrict__ output)                                      
{
  vnr::vec3i mcID(threadIdx.x + blockIdx.x * blockDim.x, 
                  threadIdx.y + blockIdx.y * blockDim.y,
                  threadIdx.z + blockIdx.z * blockDim.z);

  if (mcID.x >= mc_dims.x) return;
  if (mcID.y >= mc_dims.y) return;
  if (mcID.z >= mc_dims.z) return;

  const int mcIdx = mcID.x + mc_dims.x * (mcID.y + mc_dims.y * mcID.z);
  vnr::range1f& mc = output[mcIdx];

  if (input == 0) {
    mc.lo = 0.f;
    mc.hi = 1.f;
    return;
  }

  // compute begin/end of VOXELS for this macro-cell
  const vnr::vec3i begin = max(mcID  * mc_size - 1, vnr::vec3i(0));
  const vnr::vec3i end   = min(begin + mc_size, vol_dims);

  vnr::range1f valueRange;
  for (size_t iz = begin.z; iz < end.z; iz++)
    for (size_t iy = begin.y; iy < end.y; iy++)
      for (size_t ix = begin.x; ix < end.x; ix++) {
        const size_t pos = iz*(vol_dims.x*vol_dims.y) +
                           iy*(vol_dims.x) + ix;
        // if (pos > MAX_BYTES_SEND_TO_GPU)
        //   printf("Pos: %zu | iz: %zu | iy: %zu | ix: %zu\n", pos, iz, iy, ix);
        valueRange.extend(input[pos]);
      }

  mc.lo = valueRange.lo;
  mc.hi = valueRange.hi;
}

static size_t get_filesize(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) return 0;
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.close();
  return file_size;
}

template<typename T>
void templatedProcess(std::string vol_path, 
                      std::string& out_path, // make it an output
                      vnr::vec3i vol_dims,
                      vnr::vec3i mc_dims, 
                      vnr::vec3i mc_size) 
{
  // Create the output file
  std::string vol_dir = vol_path.substr(0, vol_path.find_last_of('/')+1);
  std::string out_filename = "macrocell_"
      + std::to_string(mc_dims.x) + 'x' + std::to_string(mc_dims.y) + 'x' + std::to_string(mc_dims.z) 
      + '_' + std::to_string(sizeof(T)) + "B.vr";
  out_path = vol_dir + out_filename;
  const size_t filesize = get_filesize(out_path);
  const size_t expected = mc_dims.long_product() * sizeof(vnr::range1f);
  if (filesize == expected) { 
    printf("[mc] Macrocell file already exists: %s\n", out_path.c_str());
    return;
  }
  printf("[mc] Generating Macrocell: %s\n", out_path.c_str());
  std::ofstream output(out_path, std::ios::binary);

  // Check if the volume file has the correct size
  size_t vol_size = get_filesize(vol_path);
  if (vol_size != vol_dims.long_product() * sizeof(T)) {
    throw std::runtime_error("[mc] Macrocell: volume size mismatch");
  }

  // Open the input volume file
  std::ifstream volume(vol_path, std::ios::in | std::ios::binary);
  if (!volume.is_open()) {
    throw std::runtime_error("[mc] Macrocell: failed to open data");
  }
  

  // Use size_t to avoid int32 overflow
  size_t encoded_bytes = sizeof(T);
  size_t mc_bytes = mc_size.x * mc_size.y * mc_size.z * encoded_bytes; // byte size of each MC
  size_t mc_count = MAX_BYTES_SEND_TO_GPU / mc_bytes; // max whole MCs we can send to the GPU
  size_t mc_sheet = vol_dims.x * vol_dims.y * mc_size.z * encoded_bytes; // byte size of 1 sheet of MCs

  T            *h_input  = new T[(mc_count * mc_bytes) / encoded_bytes];
  vnr::range1f *h_output = new vnr::range1f[mc_count];

  T            *d_input  = nullptr;
  vnr::range1f *d_output = nullptr;

  cudaMalloc(&d_input,  mc_count * mc_bytes);
  cudaMalloc(&d_output, mc_count * sizeof(vnr::range1f));

  float max_range = -MAXFLOAT, min_range = MAXFLOAT;

  vnr::vec3i read_mc_dims = mc_dims;

  for (size_t iz = 0; iz < mc_dims.z; iz += read_mc_dims.z) {
    const size_t read_pos = iz * mc_sheet;
    read_mc_dims.z = min(mc_count / (mc_dims.x*mc_dims.y), mc_dims.z - iz);

    const size_t bytes_to_read = min(read_mc_dims.z * mc_sheet, vol_size - read_pos);
    volume.seekg(read_pos);
    volume.read(reinterpret_cast<char*>(h_input), bytes_to_read);
    if (volume.gcount() != bytes_to_read) {
      throw std::runtime_error("[mc] Macrocell: Failed to read data");
    }

    cudaMemcpy(d_input, h_input, bytes_to_read, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 8, 4);
    dim3 gridSize((read_mc_dims.x + blockSize.x - 1) / blockSize.x, 
                  (read_mc_dims.y + blockSize.y - 1) / blockSize.y,
                  (read_mc_dims.z + blockSize.z - 1) / blockSize.z);
    macrocell_range<T><<<gridSize, blockSize>>>(d_input, vol_dims, read_mc_dims, mc_size, d_output);
    CUDA_SYNC_CHECK();

    const size_t read_mc = read_mc_dims.long_product();
    cudaMemcpy(h_output, d_output, read_mc * sizeof(vnr::range1f), cudaMemcpyDeviceToHost);
    for (int i=0; i<read_mc; ++i) {
      max_range = max(max_range, h_output[i].hi);
      min_range = min(min_range, h_output[i].lo);
    }

    output.write(reinterpret_cast<const char*>(h_output), read_mc * sizeof(vnr::range1f));

    printf("[mc] Macrocell %.6f%% completed\r", float(iz) / mc_dims.z * 100);
  }
  printf("[mc] Macrocell generated with MIN: %f | MAX: %f\n", min_range, max_range);

  // CleanUp
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);
  
  output.close();
  volume.close();
}

void Macrocell::process(std::string volume_path, std::string& out_path)
{   
  switch (mc_type)
  {
    case vnr::VALUE_TYPE_UINT8: templatedProcess<uint8_t>( volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_INT8: templatedProcess<int8_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_UINT16: templatedProcess<uint16_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_INT16: templatedProcess<int16_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_UINT32: templatedProcess<uint32_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_INT32: templatedProcess<int32_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_UINT64: templatedProcess<uint64_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_INT64: templatedProcess<int64_t>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_FLOAT: templatedProcess<float>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    case vnr::VALUE_TYPE_DOUBLE: templatedProcess<double>(volume_path, out_path, vol_dims, mc_dims, mc_size); break;
    default: throw std::runtime_error("Invalid mc_type");
  }
}

} // namespace tdns
