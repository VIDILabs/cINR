#pragma once

#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GcCore/libMath/Vector.hpp>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/K_Texture3dDevice.hpp>
#include <GcCore/cuda/libCommon/Surface3dDevice.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Class to handle a 3D texture on the GPU.
    *
    * @template Type of the data stored in the texture.
    */
    template<typename T>
    class Texture3dDevice : public Noncopyable, KernelObject<K_Texture3dDevice<T>>
    {

    public:
        enum AccessMode
        {
            Default = 0,
            Normalized = 1
        };

    public:

        /**
        * @brief Constructor.
        *
        * @param[in]    surface     Surface from which the texture is created.
        * @param[in]    size        Surface size (x, y, z).
        * @param[in]    elementSize Size of element (block) stored in the surface.
        * @param[in]    flag        Access mode (normalized or not).
        */
        Texture3dDevice(const tdns::common::Surface3dDevice<T> &surface,
            const tdns::math::Vector3ui &size,
            const tdns::math::Vector3ui &elementSize,
            uint32_t flag = 0);

        /**
        * @brief Destructor.
        */
        virtual ~Texture3dDevice();

        /**
        * @brief Get the texture object.
        * 
        * @return A pointer to the texture object.
        */
        cudaTextureObject_t* get_texture();

        /**
        * @brief Get the size of the texture.
        * 
        * @return A the number of elements in the 3D texture.
        */
        uint3& size();

        /**K_Texture3dDevice
        * @brief Get an object that can be send to a CUDA kernel.
        *
        * return A Texture3dDevice object that can be send to a CUDA kernel.
        */
        virtual K_Texture3dDevice<T> to_kernel_object() override;

    private:


    protected:
        /**
        * Member data.
        */
        cudaTextureObject_t _texture;       ///< The texture object.
        uint3               _size;          ///< Size of the 3D texture.
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline Texture3dDevice<T>::Texture3dDevice(const tdns::common::Surface3dDevice<T> &surface,
        const tdns::math::Vector3ui &size,
        const tdns::math::Vector3ui &elementSize,
        uint32_t flag /* = 0 */)
    {
        _size.x = size[0] * elementSize[0];
        _size.y = size[1] * elementSize[1];
        _size.z = size[2] * elementSize[2];

        // Specify texture
        struct cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = surface.get_array();

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.addressMode[2]   = cudaAddressModeClamp;
        switch(flag)
        {
            default: {
                texDesc.filterMode       = cudaFilterModePoint;
                texDesc.readMode         = cudaReadModeElementType;
                texDesc.normalizedCoords = 0;
                break;
            }
            case AccessMode::Normalized: {
                texDesc.filterMode       = cudaFilterModeLinear;
                texDesc.readMode         = (sizeof(T) > 2) ? cudaReadModeElementType : cudaReadModeNormalizedFloat; // only for 8-bit and 16-bit integer format, not for float
                texDesc.normalizedCoords = 1;
                break;
            }
        }

        CUDA_SAFE_CALL(cudaCreateTextureObject(&_texture, &resDesc, &texDesc, NULL));
    }

    //---------------------------------------------------------------------------------------------
    // Template specification for 32-bit float format : does not support cudaTextureDesc readMode "cudaReadModeNormalizedFloat"
    template<>
    inline Texture3dDevice<float1>::Texture3dDevice(const tdns::common::Surface3dDevice<float1> &surface,
        const tdns::math::Vector3ui &size,
        const tdns::math::Vector3ui &elementSize,
        uint32_t flag /* = 0 */)
    {
        _size.x = size[0] * elementSize[0];
        _size.y = size[1] * elementSize[1];
        _size.z = size[2] * elementSize[2];

        // Specify texture
        struct cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = surface.get_array();

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.addressMode[2]   = cudaAddressModeClamp;
        switch(flag)
        {
            default: {
                texDesc.filterMode       = cudaFilterModePoint;
                texDesc.readMode         = cudaReadModeElementType;
                texDesc.normalizedCoords = 0;
                break;                
            }
            case AccessMode::Normalized: {
                texDesc.filterMode       = cudaFilterModeLinear;
                texDesc.readMode         = cudaReadModeElementType;
                texDesc.normalizedCoords = 1;
                break;
            }
        }

        CUDA_SAFE_CALL(cudaCreateTextureObject(&_texture, &resDesc, &texDesc, NULL));
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    Texture3dDevice<T>::~Texture3dDevice()
    {
        cudaDestroyTextureObject(_texture);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline cudaTextureObject_t* Texture3dDevice<T>::get_texture()
    {
        return &_texture;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline uint3& Texture3dDevice<T>::size()
    {
        return _size;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline K_Texture3dDevice<T> Texture3dDevice<T>::to_kernel_object()
    {
        return K_Texture3dDevice<T>(_texture, _size);
    }

} // namespace tdns
} // namespace common