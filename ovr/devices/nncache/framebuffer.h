//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "math_def.h"

// ------------------------------------------------------------------
//
// Host Functions
//
// ------------------------------------------------------------------
#ifdef __cplusplus

#include <cuda_buffer.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ovr {
namespace nncache {

// ------------------------------------------------------------------
// OptiX Helper Functions and Classes
// ------------------------------------------------------------------

struct ISingleRB {
  virtual ~ISingleRB() = default;
  virtual void create(bool async) = 0;
  virtual void resize(size_t& count) = 0;
  virtual void download_async(cudaStream_t stream) = 0;
  virtual void* d_pointer() const = 0;
  virtual void* h_pointer() const = 0;
  virtual void deepcopy(void* dst) = 0;
  virtual void reset_buffer(cudaStream_t stream) = 0;
};

template<typename T>
struct SingleRenderBuffer : ISingleRB {
protected:
  CUDABuffer device_buffer;
  std::vector<T> host_buffer;

public:
  ~SingleRenderBuffer() override
  {
    if (device_buffer.d_pointer())
      device_buffer.free();
  }

  void create(bool async) override
  {
  }

  void resize(size_t& count) override
  {
    device_buffer.resize(count * sizeof(T));
    host_buffer.resize(count);
  }

  void download_async(cudaStream_t stream) override
  {
    device_buffer.download_async(host_buffer.data(), host_buffer.size(), stream);
  }

  void* d_pointer() const override
  {
    return (void*)device_buffer.d_pointer();
  }

  void* h_pointer() const override
  {
    return (void*)host_buffer.data();
  }

  void deepcopy(void* dst) override
  {
    std::memcpy(dst, host_buffer.data(), host_buffer.size() * sizeof(T));
  }

  void reset_buffer(cudaStream_t stream) override
  {
    CUDA_CHECK(cudaMemsetAsync((void*)device_buffer.d_pointer(), 0, device_buffer.sizeInBytes, stream));
  }
};

template<typename... Args>
struct MultipleRenderBuffers {
  cudaStream_t stream{};
  std::vector<std::shared_ptr<ISingleRB>> buffers;

  MultipleRenderBuffers()
  {
    buffers = std::vector<std::shared_ptr<ISingleRB>>{ std::make_shared<SingleRenderBuffer<Args>>()... };
  }

  ~MultipleRenderBuffers()
  {
    for (auto& b : buffers)
      b.reset();
  }

  void create(bool async)
  {
    if (async) CUDA_CHECK(cudaStreamCreate(&stream));
    for (auto& b : buffers) b->create(async);
  }

  void resize(size_t& count)
  {
    for (auto& b : buffers)
      b->resize(count);
  }

  void download_async()
  {
    for (auto& b : buffers)
      b->download_async(stream);
  }

  void* d_pointer(int layout) const
  {
    const int n = sizeof...(Args);
    if (layout >= n)
      throw std::runtime_error(std::string("cannot access the ") + std::to_string(layout) +
                               std::string("-th render buffer, as there are only ") + std::to_string(n) +
                               std::string(" render buffer(s) available."));
    return (void*)buffers[layout]->d_pointer();
  }

  void* h_pointer(int layout) const
  {
    const int n = sizeof...(Args);
    if (layout >= n)
      throw std::runtime_error(std::string("cannot access the ") + std::to_string(layout) +
                               std::string("-th render buffer, as there are only ") + std::to_string(n) +
                               std::string(" render buffer(s) available."));
    return (void*)buffers[layout]->h_pointer();
  }

  void deepcopy(int layout, void* dst)
  {
    const int n = sizeof...(Args);
    if (layout >= n)
      throw std::runtime_error(std::string("cannot access the ") + std::to_string(layout) +
                               std::string("-th render buffer, as there are only ") + std::to_string(n) +
                               std::string(" render buffer(s) available."));
    buffers[layout]->deepcopy(dst);
  }

  void reset()
  {
    for (auto& b : buffers)
      b->reset_buffer(stream);
  }
};

template<typename... Args>
struct DoubleBufferObject {
private:
  using BufferObject = MultipleRenderBuffers<Args...>;

  BufferObject& front_buffer()
  {
    return buffers[front_index];
  }

  const BufferObject& front_buffer() const
  {
    return buffers[front_index];
  }

  BufferObject& back_buffer()
  {
    return buffers[(front_index + 1) % 2];
  }

  const BufferObject& back_buffer() const
  {
    return buffers[(front_index + 1) % 2];
  }

  BufferObject buffers[2];
  int front_index{ 0 };

  size_t fb_pixel_count{ 0 };
  vec2i fb_size{ 0 };

public:
  ~DoubleBufferObject() {}

  void create(bool async = false)
  {
    buffers[0].create(async);
    buffers[1].create(async);
  }

  void resize(vec2i s)
  {
    fb_size = s;
    fb_pixel_count = (size_t)fb_size.x * fb_size.y;
    {
      buffers[0].resize(fb_pixel_count);
      buffers[1].resize(fb_pixel_count);
    }
  }

  void safe_swap()
  {
    // CUDA_CHECK(cudaStreamSynchronize(current_buffer().stream));
    front_index = (front_index + 1) % 2;
  }

  bool empty() const
  {
    return fb_pixel_count == 0;
  }

  const vec2i& size() const
  {
    return fb_size;
  }

  cudaStream_t back_stream() // stream for rendering
  {
    return back_buffer().stream;
  }

  void* front_dpointer(int layout = 0) const
  {
    return (void*)front_buffer().d_pointer(layout);
  }

  void* back_dpointer(int layout = 0) const
  {
    return (void*)back_buffer().d_pointer(layout);
  }

  void download_front()
  {
    front_buffer().download_async();
  }

  void* front_hpointer(int layout = 0) const
  {
    return (void*)front_buffer().h_pointer(layout);
  }

  void reset()
  {
    buffers[0].reset();
    buffers[1].reset();
  }
};

}
} // namespace ovr

#endif // #ifdef __cplusplus
