#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/cuda/libCommon/K_QueueDevice.hpp>

namespace tdns
{
namespace common
{
    template<typename T>
    class QueueDevice : public KernelObject<K_QueueDevice<T>>, Noncopyable
    {
    public:
        QueueDevice(size_t size = 1024);

        ~QueueDevice();

        virtual K_QueueDevice<T> to_kernel_object() override;

    private:
        T *_data;
        size_t _size;
        size_t *_begin;
        size_t *_end;
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    QueueDevice<T>::QueueDevice(size_t size /* = 1024*/)
    {
        CUDA_SAFE_CALL(cudaMalloc(&_data, size * sizeof(T)));
        CUDA_SAFE_CALL(cudaMalloc(&_begin, sizeof(size_t)));
        CUDA_SAFE_CALL(cudaMalloc(&_end, sizeof(size_t)));
        CUDA_SAFE_CALL(cudaMemset(_begin, 0, sizeof(size_t)));
        CUDA_SAFE_CALL(cudaMemset(_end, 0, sizeof(size_t)));

        _size = size;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    QueueDevice<T>::~QueueDevice()
    {
        CUDA_SAFE_CALL_NOEXCEPT(cudaFree(_data));
        CUDA_SAFE_CALL_NOEXCEPT(cudaFree(_begin));
        CUDA_SAFE_CALL_NOEXCEPT(cudaFree(_end));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    K_QueueDevice<T> QueueDevice<T>::to_kernel_object()
    {
        return K_QueueDevice<T>(_data, _size, _begin, _end);
    }

} // namespace tdns
} // namespace common