#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GcCore/cuda/libCommon/CudaError.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Kernel class to call in a kernel to use a static array.
    *
    * @template Type of the data to store.
    * @template Size of the array on the X axis.
    * @template Size of the array on the Y axis.
    * @template Size of the array on the 2 axis.
    */
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    class K_StaticArray3dDevice
    {
    private:
        /**
        * @brief Enum. Small trick to avoid to calculate the product sizeX * size Y
        *        each time and without to store a value.
        */
        enum : size_t
        {
            xyProduct = sizeX * sizeY
        };

    public:

        /**
        * @brief Constructor.
        *
        * @param Pointer to the array.
        */
        K_StaticArray3dDevice(T* data);
 
        /**
        * @brief Device getter on the data array.
        * 
        * @return A pointer on data of the 3D GPU array.
        */
        __device__ T* data();

        /**
        * @brief Device operator overload.
        * 
        * @param  position Position inside the data array.
        * 
        * @return The data at the given position in the array.
        */
        __device__ T& operator [] (const size_t position);
        __device__ const T& operator [] (const size_t position) const;

        __device__ T& operator () (const uint3 &position);
        __device__ const T& operator () (const uint3 &position) const;

        __device__ T& operator () (const size_t x, const size_t y, const size_t z);
        __device__ const T& operator () (const size_t x, const size_t y, const size_t z) const;

    protected:
        /**
        * Member data.
        */
        T *_data;  ///< Data of the 3D GPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::K_StaticArray3dDevice(T* data)
    {
        _data = data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    T* K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    T& K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator [] (const size_t position)
    { return _data[position]; }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    const T& K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator [] (const size_t position) const
    { return _data[position]; }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    T& K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const uint3 &position)
    {
        return _data[position.x + position.y * sizeX + position.z * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    const T& K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const uint3 &position) const
    {
        return _data[position.x + position.y * sizeX + position.z * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    T& K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const size_t x, const size_t y, const size_t z)
    {
        return _data[x + y * sizeX + z * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline __device__
    const T& K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        return _data[x + y * sizeX + z * xyProduct];
    }
} // namespace tdns
} // namespace common