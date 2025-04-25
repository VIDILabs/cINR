#pragma once

#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GcCore/libMath/Vector.hpp>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/K_Surface3dDevice.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Class to handle a 3D surface / texture on the GPU.
    *
    * @template Type of the data stored in the texture.
    */
    template<typename T>
    class Surface3dDevice : public Noncopyable, KernelObject<K_Surface3dDevice<T>>
    {
    public:

        /**
        * @brief Constructor.
        *
        * @param Number of elements to store in the surface.
        * @param Size of one element.
        *
        *      e.g size = (10, 1, 1) bricks, elementSize = (32, 32, 32) = the size of a brick.
        */
        Surface3dDevice(const tdns::math::Vector3ui &size, const tdns::math::Vector3ui &elementSize);

        /**
        * @brief Destructor.
        */
        virtual ~Surface3dDevice();

        /**
        * @brief Get the surface object.
        * 
        * @return A pointer to the surface object.
        */
        cudaSurfaceObject_t* get_surface();

        /**
        * @brief Get the cudaArray link to the surface object.
        * 
        * @return A pointer to the cudaArray link to the surface object.
        */
        cudaArray* get_array() const;

        /**
        * @brief Get an object that can be send to a CUDA kernel.
        *
        * return A Surface3dDevice object that can be send to a CUDA kernel.
        */
        virtual K_Surface3dDevice<T> to_kernel_object() override;

    private:

        /**
        * @brief Method to create the channel format desc given T type.
        *        It is a specialization template in order to handle
        *        the different cuda types (int3, float3, etc.).
        *        The maximum value per channel is 64 bits.
        *
        * @return Return the channel format description.
        */
        cudaChannelFormatDesc create_channel_format_desc();

    protected:
        /**
        * Member data.
        */
        cudaSurfaceObject_t _surface;       ///< The surface object.
        cudaArray           *_dataArray;    ///< Array where are stored the data.
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    Surface3dDevice<T>::Surface3dDevice(const tdns::math::Vector3ui &size, const tdns::math::Vector3ui &elementSize)
    {
        cudaChannelFormatDesc channelDesc = create_channel_format_desc();

        cudaExtent ext;
        ext.width = size[0] * elementSize[0];
        ext.height = size[1] * elementSize[1];
        ext.depth = size[2] * elementSize[2];

        CUDA_SAFE_CALL(cudaMalloc3DArray(&_dataArray, &channelDesc, ext, cudaArraySurfaceLoadStore));
        std::vector<T> vec;
        vec.resize(ext.width * ext.height * ext.depth);
        std::memset(static_cast<void*>(vec.data()), 0, vec.size() * sizeof(T));

        cudaMemcpy3DParms copyParams = {0};
        copyParams.kind = cudaMemcpyHostToDevice;
        copyParams.srcPtr = make_cudaPitchedPtr(static_cast<void*>(vec.data()), ext.width * sizeof(T), ext.width, ext.height);
        copyParams.srcPos = make_cudaPos(0, 0, 0);
        copyParams.dstArray = _dataArray;
        copyParams.dstPos = make_cudaPos(0, 0, 0);
        copyParams.extent = ext;
        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));

        // Specify texture
        cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = _dataArray;

        CUDA_SAFE_CALL(cudaCreateSurfaceObject(&_surface, &resDesc));
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    Surface3dDevice<T>::~Surface3dDevice()
    {
        CUDA_SAFE_CALL_NOEXCEPT(cudaDestroySurfaceObject(_surface));
        CUDA_SAFE_CALL_NOEXCEPT(cudaFreeArray(_dataArray));
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline cudaSurfaceObject_t* Surface3dDevice<T>::get_surface()
    {
        return &_surface;
    }
    
    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline cudaArray* Surface3dDevice<T>::get_array() const
    {
        return _dataArray;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline K_Surface3dDevice<T> Surface3dDevice<T>::to_kernel_object()
    {
        return K_Surface3dDevice<T>(_surface);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<uchar1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            8, 0, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<uchar2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            8, 8, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<uchar4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            8, 8, 8, 8,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<char1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            8, 0, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<char2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            8, 8, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<char4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            8, 8, 8, 8,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<ushort1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            16, 0, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<ushort2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            16, 16, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<ushort4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            16, 16, 16, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<short1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            16, 0, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<short2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            16, 16, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<short4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            16, 16, 16, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<uint1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 0, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<uint2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<uint4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 32, 32,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<int1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 0, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<int2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<int4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 32, 32,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<ulonglong1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 0, 0,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<ulonglong2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 32, 32,
            cudaChannelFormatKindUnsigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<longlong1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 0, 0,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<longlong2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 32, 32,
            cudaChannelFormatKindSigned);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<float1>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 0, 0, 0,
            cudaChannelFormatKindFloat);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<float2>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 0, 0,
            cudaChannelFormatKindFloat);
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline cudaChannelFormatDesc Surface3dDevice<float4>::create_channel_format_desc()
    {
        return cudaCreateChannelDesc(
            32, 32, 32, 32,
            cudaChannelFormatKindFloat);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline cudaChannelFormatDesc Surface3dDevice<T>::create_channel_format_desc()
    {
        // e.g float4 : 32, 32, 32, 32; int2 : 32, 32, 0, 0
        return cudaCreateChannelDesc(
            8 * sizeof(T), 0, 0, 0,
            cudaChannelFormatKindUnsigned);
    }
} // namespace tdns
} // namespace common