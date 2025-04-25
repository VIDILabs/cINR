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

    namespace StaticArrayOptions
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
    }// StaticArrayOptions
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
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option = 0>
    class StaticArray3dHost : public Noncopyable
    {
    public:

        /**
        * @brief Default constructor.
        */
        StaticArray3dHost();
 
        /**
        * @brief Destructor.
        */
        ~StaticArray3dHost();

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

        /**
        * @brief Overload assignment operator.
        */
        StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>& operator = (
            StaticArray3dDevice<T, sizeX, sizeY, sizeZ> &srcArray);

        StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>& operator = (
            StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option> &srcArray);

        StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>& operator = (
            T* srcArray);

    protected:
        /**
        * Member data.
        */
        T *_data;  ///< Data of the 3D CPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::StaticArray3dHost()
    {
        _data = nullptr;
        switch (Option)
        {
        case StaticArrayOptions::Options::Pinned:
            // Allocation of the array on the CPU with CUDA using pinned memory
            CUDA_SAFE_CALL(cudaMallocHost(
                reinterpret_cast<void**> (&_data),
                sizeX * sizeY * sizeZ * sizeof(T)));
            break;

        case StaticArrayOptions::Options::Mapped:
            // Allocation of the array on the CPU with CUDA using GPU mapped memory
            CUDA_SAFE_CALL(cudaHostAlloc(
                reinterpret_cast<void**> (&_data),
                sizeX * sizeY * sizeZ * sizeof(T),
                cudaHostAllocMapped | cudaHostAllocWriteCombined));
            break;

        case StaticArrayOptions::Options::Standard:
            // Allocation of the array on the CPU in a classic way
            _data = new T[sizeX * sizeY * sizeZ];
            break;
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::~StaticArray3dHost()
    {
        switch (Option)
        {
        case StaticArrayOptions::Options::Pinned:
        case StaticArrayOptions::Options::Mapped:
            CUDA_SAFE_CALL(cudaFreeHost( _data));
            break;
        case StaticArrayOptions::Options::Standard:
            delete[] _data;
            break;
        }
        _data = nullptr;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline T* StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline cudaPitchedPtr StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::get_cuda_pitched_ptr()
    {
        return make_cudaPitchedPtr(
            reinterpret_cast<void *> (_data),
            static_cast<size_t>(sizeX) * sizeof(T),
            static_cast<size_t>(sizeX),
            static_cast<size_t>(sizeY));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline T& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator [] (const size_t position)
    {
        return _data[position];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline const T& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator [] (const size_t position) const
    {
        return _data[position];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline T& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator () (const tdns::math::Vector3ui &position)
    {
        return _data[position[0] + position[1] * sizeX + position[2] * sizeX * sizeY];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline const T& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator () (const tdns::math::Vector3ui &position) const
    {
        return _data[position[0] + position[1] * sizeX + position[2] * sizeX * sizeY];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline T& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator () (const size_t x, const size_t y, const size_t z)
    {
        return _data[x + y * sizeX + z * sizeX * sizeY];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline const T& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        return _data[x + y * sizeX + z * sizeX * sizeY];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator = (
        StaticArray3dDevice<T, sizeX, sizeY, sizeZ> &srcArray)
    {
        CUDA_SAFE_CALL(cudaMemcpy(  _data,
                                    srcArray.data(),
                                    sizeX * sizeY * sizeZ * sizeof(T),
                                    cudaMemcpyDeviceToHost));
        return *this;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator = (
        StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option> &srcArray)
    {
        switch (Option)
        {
        case StaticArrayOptions::Options::Pinned:
        case StaticArrayOptions::Options::Mapped:
            CUDA_SAFE_CALL(cudaMemcpy(_data,
                srcArray.data(),
                sizeX * sizeY * sizeZ * sizeof(T),
                cudaMemcpyHostToHost));
            break;
        case StaticArrayOptions::Options::Standard:
            memcpy(_data, srcArray.data(), sizeX * sizeY * sizeZ * sizeof(T));
            break;
        }
        return *this;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    inline StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>& StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option>::operator = (
        T* srcArray)
    {
        switch (Option)
        {
        case StaticArrayOptions::Options::Pinned:
        case StaticArrayOptions::Options::Mapped:
            CUDA_SAFE_CALL(cudaMemcpy(_data,
                srcArray,
                sizeX * sizeY * sizeZ * sizeof(T),
                cudaMemcpyHostToHost));
            break;
        case StaticArrayOptions::Options::Standard:
            memcpy(_data, srcArray, sizeX * sizeY * sizeZ * sizeof(T));
            break;
        }
        return *this;
    }
} // namespace tdns
} // namespace common

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline void memcpy_array(T* hostDstArray, StaticArray3dDevice<T, sizeX, sizeY, sizeZ>* srcArray, uint32_t numElems)
    {
        CUDA_SAFE_CALL(cudaMemcpy(  hostDstArray,
                                    srcArray->data(),
                                    numElems * sizeof(T),
                                    cudaMemcpyDeviceToHost));
    }
} // namespace tdns
} // namespace common