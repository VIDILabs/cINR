#pragma once

#include <cstdint>

#include <GcCore/libMath/Vector.hpp>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/cuda/libCommon/StaticArray3dDevice.hpp>

namespace tdns
{
namespace common
{

    namespace DynamicArrayOptions
    {        
        /**
        * @brief Enum used for the array option.
        */
        enum Options
        {
            Pinned = 0,     // Pinned (non-paginable) memory allocation
            Mapped,         // GPU mapped (into CUDA address space) memory allocation
            Standard        // Classic CPU allocation
        };
    }// DynamicArrayOptions
    /**
    * @brief Class to create a static array on the CPU given cuda option.
    *
    * @template Type of the data to store.
    * @template Size on of the array on X axis.
    * @template Size on of the array on Y axis.
    * @template Size on of the array on Z axis.
    * @template Option of the array. How the allocation is performed.
    *           The default value is a pinned memory allocation.
    */
    template<typename T, uint32_t Option = 0>
    class DynamicArray3dHost : public Noncopyable
    {
    public:

        /**
        * @brief Default constructor.
        */
        DynamicArray3dHost(const tdns::math::Vector3ui &size);
 
        /**
        * @brief Destructor.
        */
        ~DynamicArray3dHost();

        /**
        * @brief Getter on the data array.
        * 
        * @return A pointer on data of the 3D host array.
        */
        T* data();

        /**
        * @brief Get a CUDA Pitched memory pointer.
        * 
        * @return A CUDA Pitched memory pointer.
        */
        cudaPitchedPtr get_cuda_pitched_ptr();

        /**
        * @brief
        * 
        * @param[out]
        * @return
        */
        void get_mapped_device_pointer(T **d_data);

        /**
        * @brief Operator overload.
        * 
        * @param  position Position inside the data array.
        * 
        * @return The data at the given position in the array.
        */
        T& operator [] (const size_t position);
        const T& operator [] (const size_t position) const;

        T& operator () (const tdns::math::Vector3ui &position);
        const T& operator () (const tdns::math::Vector3ui &position) const;

        T& operator () (const size_t x, const size_t y, const size_t z);
        const T& operator () (const size_t x, const size_t y, const size_t z) const;

    protected:
        /**
        * Member data.
        */
        uint32_t                _xyProduct; ///< The product of the x and y size of the array.
        tdns::math::Vector3ui   _size;      ///<
        T                       *_data;     ///< Data of the 3D CPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline DynamicArray3dHost<T, Option>::DynamicArray3dHost(const tdns::math::Vector3ui &size)
    {
        _data = nullptr;
        _size = size;
        _xyProduct = size[0] * size[1];

        switch (Option)
        {
        case DynamicArrayOptions::Options::Pinned:
            // Allocation of the array on the CPU with CUDA using pinned memory
            CUDA_SAFE_CALL(cudaMallocHost(
                reinterpret_cast<void**> (&_data),
                size[0] * size[1] * size[2] * sizeof(T)));
            break;

        case DynamicArrayOptions::Options::Mapped:
            // Allocation of the array on the CPU with CUDA using GPU mapped memory
            CUDA_SAFE_CALL(cudaHostAlloc(
                reinterpret_cast<void**> (&_data),
                size[0] * size[1] * size[2] * sizeof(T),
                cudaHostAllocMapped | cudaHostAllocWriteCombined));
            break;

        case DynamicArrayOptions::Options::Standard:
            // Allocation of the array on the CPU in a classic way
            _data = new T[size[0] * size[1] * size[2]];
            break;
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline DynamicArray3dHost<T, Option>::~DynamicArray3dHost()
    {
        switch (Option)
        {
        case DynamicArrayOptions::Options::Pinned:
        case DynamicArrayOptions::Options::Mapped:
            CUDA_SAFE_CALL_NOEXCEPT(cudaFreeHost(_data));
            break;
        case DynamicArrayOptions::Options::Standard:
            delete[] _data;
            break;
        }
        _data = nullptr;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline T* DynamicArray3dHost<T, Option>::data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline cudaPitchedPtr DynamicArray3dHost<T, Option>::get_cuda_pitched_ptr()
    {
        return make_cudaPitchedPtr(
            reinterpret_cast<void *> (_data),
            static_cast<size_t>(_size[0]) * sizeof(T),
            static_cast<size_t>(_size[0]),
            static_cast<size_t>(_size[1]));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline void DynamicArray3dHost<T, Option>::get_mapped_device_pointer(T **d_data)
    {
        cudaHostGetDevicePointer(d_data, _data, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline T& DynamicArray3dHost<T, Option>::operator [] (const size_t position)
    {
        return _data[position];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline const T& DynamicArray3dHost<T, Option>::operator [] (const size_t position) const
    {
        return _data[position];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline T& DynamicArray3dHost<T, Option>::operator () (const tdns::math::Vector3ui &position)
    {
        return _data[position[0] + position[1] * _size[0] + position[2] * _xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline const T& DynamicArray3dHost<T, Option>::operator () (const tdns::math::Vector3ui &position) const
    {
        return _data[position[0] + position[1] * _size[0] + position[2] * _xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline T& DynamicArray3dHost<T, Option>::operator () (const size_t x, const size_t y, const size_t z)
    {
        return _data[x + y * _size[0] + z * _xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t Option>
    inline const T& DynamicArray3dHost<T, Option>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        return _data[x + y * _size[0] + z * _xyProduct];
    }
} // namespace tdns
} // namespace common