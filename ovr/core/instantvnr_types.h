#ifndef INSTANT_VNR_TYPES_H
#define INSTANT_VNR_TYPES_H

// #include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/cuda_buffer.h>

#include <gdt/random/random.h>

#include "mathdef.h"
#include "array.h"

#ifdef CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS
#define VNR_VERBOSE_MEMORY_ALLOCS
#endif
// #define VNR_VERBOSE_MEMORY_ALLOCS

#ifdef NDEBUG
#define TRACE_CUDA ((void)0)
#else
#define TRACE_CUDA CUDA_SYNC_CHECK()
#endif

#ifdef NDEBUG
#define ASSERT_THROW(X, MSG) ((void)0)
#else
#define ASSERT_THROW(X, MSG) { if (!(X)) throw std::runtime_error(MSG); }
#endif

#define INSTANT_VNR_NAMESPACE_BEGIN namespace vnr {
#define INSTANT_VNR_NAMESPACE_END }

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

INSTANT_VNR_NAMESPACE_BEGIN

struct Camera {
public:
  /*! camera position - *from* where we are looking */
  vec3f from;
  /*! which point we are looking *at* */
  vec3f at;
  /*! general up-vector */
  vec3f up;
  /*! fovy in degrees */
  float fovy = 60;
};

// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------

using RandomTEA = gdt::LCG<16>;

// ------------------------------------------------------------------
// Additional Kernel Helper Functions
// ------------------------------------------------------------------
#ifdef __CUDACC__

// template <typename T>
// __forceinline__ __device__ T lerp(float r, const T& a, const T& b) 
// {
//   return (1-r) * a + r * b;
// }

// __forceinline__ __device__ bool
// block_any(bool v)
// {
//   return __syncthreads_or(v);
// }

// ------------------------------------------------------------------
// Additional Atomic Functions
// ------------------------------------------------------------------
// reference: https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250

__forceinline__ __device__ float
atomicMin(float* addr, float value)
{
  float old;
  old = !signbit(value) ? __int_as_float(::atomicMin((int*)addr, __float_as_int(value))) : __uint_as_float(::atomicMax((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

__forceinline__ __device__ float
atomicMax(float* addr, float value)
{
  float old;
  old = !signbit(value) ? __int_as_float(::atomicMax((int*)addr, __float_as_int(value))) : __uint_as_float(::atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

#endif // __CUDACC__

// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------

struct TransferFunction {
public:
  std::vector<vec3f> color;
  std::vector<vec2f> alpha;
  range1f range;
};

struct DeviceTransferFunction {
  Array1DFloat4 colors;
  Array1DScalar alphas;
  range1f range;
  float range_rcp_norm;
};

struct TransferFunctionObject {
  DeviceTransferFunction tfn;
  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};
  ~TransferFunctionObject() { clean(); }
  void clean();
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r, cudaStream_t stream);
  void update(const TransferFunction& tfn, /*const range1f original_data_range,*/ cudaStream_t stream);
};

// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------

struct VolumeDesc {
public:
  struct File {
    std::string filename;
    size_t offset;
    size_t nbytes;
    bool bigendian;
    void* rawptr{ nullptr };
  };
public:
  vec3i     dims;
  ValueType type;
  range1f   range;
  vec3f scale = 0;
  vec3f translate = 0;
  std::vector<File> data;
};

struct VolumeObject 
{
  virtual const cudaTextureObject_t& texture()  const = 0;
  virtual ValueType get_data_type()             const = 0;
  // virtual range1f   get_data_value_range()      const = 0;
  virtual vec3i     get_data_dims()             const = 0;
  virtual affine3f  get_data_transform()        const = 0;
  virtual float*    get_macrocell_max_opacity() const = 0;
  virtual vec2f*    get_macrocell_value_range() const = 0;
  virtual vec3i     get_macrocell_dims()        const = 0;
  virtual vec3f     get_macrocell_spacings()    const = 0;
  virtual void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) = 0;
  virtual void set_data_transform(affine3f transform) = 0;
};

INSTANT_VNR_NAMESPACE_END

#endif // INSTANT_VNR_TYPES_H
