#pragma once

#include <cstdint>
#include <cuda.h>

#include <GcCore/libMath/Vector.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Kernel class to call in a kernel to use a 3D array.
    *        
    * @template Type of data to store.
    */
    template<typename T>
    class K_DynamicArray3dDevice
    {
    public:

        /**
        * @brief Force creation of the default constructor.
        */
        K_DynamicArray3dDevice() = default;
        
        /**
        * @brief Create the object from existing data.
        *
        * @param Pointer to the array.
        * @param 3D size of the array.
        */
        K_DynamicArray3dDevice(T *data, const tdns::math::Vector3ui &size);
 
        /**
        * @brief Destructor.
        */
        ~K_DynamicArray3dDevice() = default;

        /**
        * @brief Device getter on the data array.
        * 
        * @return A pointer on data of the 3D GPU array.
        */
        __device__ T* data();

        /**
         * @brief
         */
        __device__ size_t size() const;

        /**
         * @brief
         */
        __device__ uint3 dims() const;

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
        uint32_t    _xyProduct; ///< The product of the x and y size of the array.
        uint3       _size;      ///< 3D size of the array.
        T           *_data;     ///< Data of the 3D GPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    K_DynamicArray3dDevice<T>::K_DynamicArray3dDevice(T *data, const tdns::math::Vector3ui &size)
    {
        _xyProduct = size[0] * size[1];

        _size = *reinterpret_cast<const uint3*>(size.data());

        _data = data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T* K_DynamicArray3dDevice<T>::data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    size_t K_DynamicArray3dDevice<T>::size() const
    {
        return static_cast<size_t>(_size.x) * static_cast<size_t>(_size.y) * static_cast<size_t>(_size.z);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    uint3 K_DynamicArray3dDevice<T>::dims() const
    {
        return _size;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_DynamicArray3dDevice<T>::operator [] (const size_t idx)
    { 
#if TDNS_MODE == TDNS_MODE_DEBUG
        const size_t len = size();
        if (idx >= len) { printf("[K_DArray3D[size_t]] Index (%d) Out-of-Bound (%d)\n", (uint32_t)idx, (uint32_t)len); asm("trap;"); }
#endif
        return _data[idx]; 
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_DynamicArray3dDevice<T>::operator [] (const size_t idx) const
    {
#if TDNS_MODE == TDNS_MODE_DEBUG
        const size_t len = size();
        if (idx >= len) { printf("[K_DArray3D[size_t]] Index (%d) Out-of-Bound (%d)\n", (uint32_t)idx, (uint32_t)len); asm("trap;"); }
#endif
        return _data[idx]; 
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_DynamicArray3dDevice<T>::operator () (const uint3 &position)
    {
        const size_t idx = static_cast<size_t>(position.x) + static_cast<size_t>(position.y) * _size.x + static_cast<size_t>(position.z) * _xyProduct;
#if TDNS_MODE == TDNS_MODE_DEBUG
        const size_t len = size();
        if (idx >= len) { printf("[K_DArray3D(uint3)] Index (%d) Out-of-Bound (%d)\n", (uint32_t)idx, (uint32_t)len); asm("trap;"); }
#endif
        return _data[idx];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_DynamicArray3dDevice<T>::operator () (const uint3 &position) const
    {
        const size_t idx = static_cast<size_t>(position.x) + static_cast<size_t>(position.y) * _size.x + static_cast<size_t>(position.z) * _xyProduct;
#if TDNS_MODE == TDNS_MODE_DEBUG
        const size_t len = size();
        if (idx >= len) { printf("[K_DArray3D(uint3)] Index (%d) Out-of-Bound (%d)\n", (uint32_t)idx, (uint32_t)len); asm("trap;"); }
#endif
        return _data[idx];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_DynamicArray3dDevice<T>::operator () (const size_t x, const size_t y, const size_t z)
    {
        const size_t idx = x + y * _size.x + z * _xyProduct;
#if TDNS_MODE == TDNS_MODE_DEBUG
        const size_t len = size();
        if (idx >= len) { printf("[K_DArray3D(int,int,int)] Index (%d) Out-of-Bound (%d)\n", (uint32_t)idx, (uint32_t)len); asm("trap;"); }
#endif
        return _data[idx];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_DynamicArray3dDevice<T>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        const size_t idx = x + y * _size.x + z * _xyProduct;
#if TDNS_MODE == TDNS_MODE_DEBUG
        const size_t len = size();
        if (idx >= len) { printf("[K_DArray3D(int,int,int)] Index (%d) Out-of-Bound (%d)\n", (uint32_t)idx, (uint32_t)len); asm("trap;"); }
#endif
        return _data[idx];
    }
} // namespace tdns
} // namespace common