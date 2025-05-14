#pragma once

#include "../launchparams.h"

#ifdef __CUDACC__
#include <cub/cub.cuh> 
#endif

#include <functional>

namespace ovr::nncache {

typedef std::function<void(cudaStream_t, uint32_t, vec3f* __restrict__, float* __restrict__)> IterativeSampler;

class MethodRayMarching {
public:
  enum ShadingMode { NO_SHADING = 0, GRADIENT_SHADING, SINGLE_SHADE_HEURISTIC, SHADOW };

  ~MethodRayMarching() { clear(0); }
  void render(cudaStream_t stream, const LaunchParams& params, const VNRDeviceVolume& volume, const IterativeSampler& sampler, ShadingMode mode, bool iterative = false);
  void clear(cudaStream_t stream) { sample_streaming_buffer.free(stream); }

private:
  CUDABuffer sample_streaming_buffer;
};

class MethodPathTracing {
public:

  ~MethodPathTracing() { clear(0); }
  void render(cudaStream_t stream, const LaunchParams& params, const VNRDeviceVolume& volume, const IterativeSampler& sampler, bool iterative = false);
  void clear(cudaStream_t stream) { 
    samples_buffer.free(stream); 
    packets_buffer.free(stream); 
    counter_buffer.free(stream); 
  }

private:
  CUDABuffer samples_buffer;
  CUDABuffer packets_buffer;
  CUDABuffer counter_buffer;
};

class MethodMegakernel {
public:
  void render(cudaStream_t stream, LaunchParams& params, DeviceGrid& grid, OpaqueCacheManager& cache, bool reset_accumulation);
  void clear(cudaStream_t stream) { params_buffer.free(stream); accumulation_buffer.free(stream); }

private:
  CUDABuffer params_buffer;
  CUDABuffer accumulation_buffer;
};


// ------------------------------------------------------------------------------
// Device Functions
// ------------------------------------------------------------------------------

#ifdef __CUDACC__

template<typename ItemsType>
union __align__(16) PacketTemplate
{
  typedef uint4 DataType;
  static constexpr int N = (sizeof(ItemsType) + sizeof(DataType) - 1) / sizeof(DataType);

  uint4 data[N];
  ItemsType items;

  __device__ PacketTemplate() {}

  __device__ PacketTemplate(const ItemsType& items) : items(items) {}

  __device__ PacketTemplate& operator=(const PacketTemplate& other) {
    if (this == &other) return *this;
    items = other.items;
    return *this;
  }
};

template<
  typename SoAData, typename SoAView,
  typename OffsetT = ptrdiff_t
>
class SoAIterator {
public:
  // Required iterator traits
  typedef SoAIterator  self_type;  ///< My own type
  typedef OffsetT difference_type; ///< Type to express the result of subtracting one iterator from another
  typedef SoAData   value_type;  ///< The type of the element the iterator can point to
  typedef void      pointer;     ///< The type of a pointer to an element the iterator can point to
  typedef SoAView   reference;   ///< The type of a reference to an element the iterator can point to
  typedef std::random_access_iterator_tag iterator_category;  ///< The iterator category

private:
  SoAView view;

public:
  /// Constructor
  __host__ __device__ __forceinline__ 
  SoAIterator(const SoAView view, OffsetT offset = 0) : view(view) {
    this->view.move(offset);
  }

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    view.move(1);
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    view.move(1);
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const {
    return view;
  }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(view, n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type& operator+=(Distance n) {
    view.move(n);
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(view, - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type& operator-=(Distance n) {
    view.move(-n);
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_type other) const {
    return view.diff(other);
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    self_type offset = (*this) + n;
    return *offset;
  }
};

template<
  typename SoAData, typename SoAView, typename OffsetT = ptrdiff_t
>
inline uint32_t 
inplace_compaction(cudaStream_t stream, uint32_t& num_items, uint8_t* d_flags, const SoAView& data)
{
  // Declare, allocate, and initialize device-accessible pointers for input, flags, and output
  SoAIterator<SoAData, SoAView, OffsetT> d_data(data);
  uint32_t *d_num_selected_out;
  CUDA_CHECK(cudaMallocAsync((void**)&d_num_selected_out, sizeof(uint32_t), stream));

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(
    d_temp_storage, temp_storage_bytes,
    d_data, d_flags, d_num_selected_out, 
    num_items, stream);

  // Allocate temporary storage
  CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

  // Run selection
  cub::DeviceSelect::Flagged(
    d_temp_storage, temp_storage_bytes,
    d_data, d_flags, d_num_selected_out, 
    num_items, stream);

  // Cleanup
  CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
  CUDA_CHECK(cudaMemsetAsync(d_flags, 0, num_items * sizeof(uint8_t), stream));
  CUDA_CHECK(cudaMemcpyAsync(&num_items, d_num_selected_out, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_num_selected_out, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return num_items;
}

#endif

}
