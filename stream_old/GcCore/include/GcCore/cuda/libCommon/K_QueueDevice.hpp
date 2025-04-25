#pragma once

#include "cuda_runtime.h"

namespace tdns
{
namespace common
{
    template<typename T>
    class K_QueueDevice
    {
    public:
        K_QueueDevice(T* data, size_t _size, size_t *_begin, size_t *_end);

        __device__ T& operator [] (size_t i);
        __device__ const T& operator [] (size_t i) const;

        __device__ bool empty() const;

        __device__ size_t size() const;

        __device__ bool push(const T &value);

        __device__ bool pop(T &value);

    private:
        T *_data;
        size_t _size;
        size_t *_begin;
        size_t *_end;
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    K_QueueDevice<T>::K_QueueDevice(T* data, size_t size, size_t *begin, size_t *end)
    {
        _data = data;
        _size = size;
        _begin = begin;
        _end = end;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_QueueDevice<T>::operator [] (size_t i)
    {
        return _data[(*_begin + i) % _size];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_QueueDevice<T>::operator [] (size_t i) const
    {
        return _data[(*_begin + i) % _size];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    bool K_QueueDevice<T>::empty() const
    {
        return (*_end == *_begin);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    size_t K_QueueDevice<T>::size() const
    {
        return *_end - *_begin;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    bool K_QueueDevice<T>::push(const T &value)
    {
        const size_t &begin = *_begin;
        size_t &end = *_end;
        if (end + 1 == begin) return false;

        _data[end] = value;
        end + 1 >= _size ? end = 0 : ++end;

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    bool K_QueueDevice<T>::pop(T &value)
    {
        size_t &begin = *_begin;
        const size_t &end = *_end;
        if (begin == end) return false;

        value = _data[begin];
        begin + 1 >= _size ? begin = 0 : ++begin;

        return true;
    }
} // namespace tdns
} // namespace common