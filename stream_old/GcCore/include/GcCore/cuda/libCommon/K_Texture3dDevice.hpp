#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace tdns
{
namespace common
{
    /**
    * @brief Kernel class to call in a kernel to use a texture.
    *
    * @template Type of the data stored in the texture.
    */
    template<typename T>
    class K_Texture3dDevice
    {
    public:

        /**
        * @brief Constructor.
        *
        * @param The texture this object will refere to.
        * It does not become the owner of it.
        */
        K_Texture3dDevice(cudaTextureObject_t texture, const uint3 &size);

        /**
        * @brief Get the size of the texture.
        * 
        * @return A the number of elements in the 3D texture.
        */
        __device__ const uint3& size() const;

        /**
        * @brief Device getter on the texture memory.
        * 
        * @param The 3D floating point position inside the texture.
        * 
        * @return The data at the given 3D position inside the texture.
        */
        __device__ T get(const uint3 &position) const;

        /**
        * @brief Device getter on the texture memory.
        * 
        * @param The 3D floating point position inside the texture.
        * 
        * @return The data at the given 3D position inside the texture.
        */
        template<typename U>
        __device__ U get_normalized(const float3 &position) const;

    protected:
        /**
        * Member data.
        */
        cudaTextureObject_t _texture;   ///< The texture object.
        uint3               _size;      ///< Size of the 3D texture.        
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    K_Texture3dDevice<T>::K_Texture3dDevice(cudaTextureObject_t texture, const uint3 &size)
    {
        _texture = texture;
        _size = size;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const uint3& K_Texture3dDevice<T>::size() const
    {
        return _size;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T K_Texture3dDevice<T>::get(const uint3 &position) const
    {
        return tex3D<T>(_texture, position.x, position.y, position.z);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    template<typename U>
    inline __device__
    U K_Texture3dDevice<T>::get_normalized(const float3 &position) const
    {
        return tex3D<U>(_texture, position.x, position.y, position.z);
    }

} // namespace tdns
} // namespace common