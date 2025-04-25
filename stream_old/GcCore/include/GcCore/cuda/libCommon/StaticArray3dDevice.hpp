#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GcCore/libMath/Vector.hpp>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/K_StaticArray3dDevice.hpp>

// Forward declarations.
namespace tdns
{
namespace common
{
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t Option>
    class StaticArray3dHost;
} // namespace tdns
} // namespace common

namespace tdns
{
namespace common
{
    /**
    * @brief Class that create a 3D array on the GPU.
    *
    * @template Type of the data to store.
    * @template Size of the array on the X axis.
    * @template Size of the array on the Y axis.
    * @template Size of the array on the 2 axis.
    */
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    class StaticArray3dDevice : public KernelObject<K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>>, Noncopyable
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
        * @brief Default constructor.
        */
        StaticArray3dDevice();
 
        /**
        * @brief Destructor.
        */
        virtual ~StaticArray3dDevice();

        /**
        * @brief Getter on the data array.
        * 
        * @return A pointer on data of the 3D GPU array.
        */
        T* data();

        /**
        * @brief Get an object that can be send to a CUDA kernel.
        * 
        * @return A StaticArray3dDevice object that can be send to a CUDA kernel.
        */
        virtual K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ> to_kernel_object() override;

        /**
        * @brief Get a CUDA Pitched memory pointer.
        * 
        * @return A CUDA Pitched memory pointer.
        */
        cudaPitchedPtr get_cuda_pitched_ptr();

        /**
        * @brief Operator overload.
        * 
        * @param Position inside the data array.
        * 
        * @return The data at the given position in the array.
        */
        T* operator [] (const size_t position);
        const T* operator [] (const size_t position) const;

        T* operator () (const tdns::math::Vector3ui &position);
        const T* operator () (const tdns::math::Vector3ui &position) const;

        T* operator () (const size_t x, const size_t y, const size_t z);
        const T* operator () (const size_t x, const size_t y, const size_t z) const;

        /**
        * @brief Overload assignment operator.
        */
        template<uint32_t Option>
        StaticArray3dDevice<T, sizeX, sizeY, sizeZ>& operator = (
            StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option> &srcArray);

        StaticArray3dDevice<T, sizeX, sizeY, sizeZ>& operator = (
            StaticArray3dDevice<T, sizeX, sizeY, sizeZ> &srcArray);

    protected:
        /**
        * Member data.
        */
        T *_data;   ///< Data of the 3D GPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::StaticArray3dDevice()
    {
        // Allocation of the array on the GPU with CUDA
        CUDA_SAFE_CALL(cudaMalloc(
            reinterpret_cast<void**> (&_data),
            sizeX * sizeY * sizeZ * sizeof(T)));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::~StaticArray3dDevice()
    {
        CUDA_SAFE_CALL(cudaFree( _data));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ> StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::to_kernel_object()
    {
        return K_StaticArray3dDevice<T, sizeX, sizeY, sizeZ>(_data);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline cudaPitchedPtr StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::get_cuda_pitched_ptr()
    {
        return make_cudaPitchedPtr(
            reinterpret_cast<void *> (_data),
            static_cast<size_t>(sizeX) * sizeof(T),
            static_cast<size_t>(sizeX),
            static_cast<size_t>(sizeY));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator [] (const size_t position)
    { return &_data[position]; }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline const T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator [] (const size_t position) const
    { return &_data[position]; }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const tdns::math::Vector3ui &position)
    {
        return &_data[position[0] + position[1] * sizeX + position[2] * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline const T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const tdns::math::Vector3ui &position) const
    {
        return &_data[position[0] + position[1] * sizeX + position[2] * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const size_t x, const size_t y, const size_t z)
    {
        return &_data[x + y * sizeX + z * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline const T* StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        return &_data[x + y * sizeX + z * xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    template<uint32_t Option>
    inline StaticArray3dDevice<T, sizeX, sizeY, sizeZ>& StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator = (
        StaticArray3dHost<T, sizeX, sizeY, sizeZ, Option> &srcArray)
    {
        CUDA_SAFE_CALL(cudaMemcpy(  _data,
                                    srcArray.data(),
                                    sizeX * sizeY * sizeZ * sizeof(T),
                                    cudaMemcpyHostToDevice));
        return *this;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ>
    inline StaticArray3dDevice<T, sizeX, sizeY, sizeZ>& StaticArray3dDevice<T, sizeX, sizeY, sizeZ>::operator = (
        StaticArray3dDevice<T, sizeX, sizeY, sizeZ> &srcArray)
    {
        CUDA_SAFE_CALL(cudaMemcpy(  _data,
                                    srcArray.data(),
                                    sizeX * sizeY * sizeZ * sizeof(T),
                                    cudaMemcpyDeviceToDevice));
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
    inline void memcpy_array(StaticArray3dDevice<T, sizeX, sizeY, sizeZ>* dstArray, T* hostSrcPtr, uint32_t numElems)
    {
        CUDA_SAFE_CALL(cudaMemcpy(  dstArray->data(),
                                    hostSrcPtr,
                                    numElems * sizeof(T),
                                    cudaMemcpyHostToDevice));
    }
} // namespace tdns
} // namespace common