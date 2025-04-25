#pragma once

#include <cuda.h>

#include <GcCore/cuda/libCommon/Surface3dDevice.hpp>
#include <GcCore/cuda/libCommon/Texture3dDevice.hpp>
#include <GcCore/cuda/libCommon/K_DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dDevice.hpp>

namespace tdns
{
namespace gpucache
{
    /**
    * @brief
    */
    template<typename T>
    class K_MultiResolutionPageDirectory
    {
    public:

        /**
        * @brief Constructor.
        * 
        * @param
        * @param
        * @param
        */
        K_MultiResolutionPageDirectory(
            tdns::common::Surface3dDevice<T> &data,
            tdns::common::Texture3dDevice<T> &texture,
            tdns::common::DynamicArray3dDevice<uint32_t> &levelCoordinates, 
            tdns::common::DynamicArray3dDevice<uint3> &levelDimensions);

        /**
        * @brief Getters / Setters
        * 
        * @return The data.
        */
        __device__ tdns::common::K_Surface3dDevice<T>* data();

        /**
         * @brief
         */
        __device__ tdns::common::K_DynamicArray3dDevice<uint3>& get_level_dimensions();

        /**
         * @brief
         */
        __device__ tdns::common::K_DynamicArray3dDevice<uint32_t>& get_level_coordinates();

        /**
         * @brief
         */
        __device__ uint32_t get_nb_levels_of_resolution();

        __device__ uint3 get_coords(uint32_t level, const float3 &position) const;
        __device__ T get(const uint3 &surfCoord) const;

        /**
        * [get description]
        * @param  level    [description]
        * @param  position [description]
        * @return          [description]
        */
        __device__ T get(uint32_t level, const float3 &position) const;

        /**
        * [set description]
        * @param position [description]
        * @param value    [description]
        */
        __device__ void set(const uint3 &position, const T &value);
        
        /**
        * [set description]
        * @param level    [description]
        * @param position [description]
        * @param value    [description]
        */
        __device__ void set(uint32_t level, const float3 &position, const T &value) const;

    protected:
        /**
        * Member data.
        */
        tdns::common::K_DynamicArray3dDevice<uint32_t>  _levelCoordinates;  ///< 3D positions of the first entry of each level of resolution.
        tdns::common::K_DynamicArray3dDevice<uint3>     _levelDimensions;   ///< 3D sizes of each level.
        tdns::common::K_Surface3dDevice<T>              _data;              ///< Addresses in a surface.
        tdns::common::K_Texture3dDevice<T>              _texture;           ///< Addresses in a texture.

    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    K_MultiResolutionPageDirectory<T>::K_MultiResolutionPageDirectory(
        tdns::common::Surface3dDevice<T> &data,
        tdns::common::Texture3dDevice<T> &texture,
        tdns::common::DynamicArray3dDevice<uint32_t> &levelCoordinates,
        tdns::common::DynamicArray3dDevice<uint3> &levelDimensions)
        : _data(data.to_kernel_object()),
        _texture(texture.to_kernel_object())
    {
        _levelCoordinates = levelCoordinates.to_kernel_object();
        _levelDimensions = levelDimensions.to_kernel_object();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    tdns::common::K_Surface3dDevice<T>* K_MultiResolutionPageDirectory<T>::data()
    {
        return &_data;
    }
    
    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    tdns::common::K_DynamicArray3dDevice<uint3>& K_MultiResolutionPageDirectory<T>::get_level_dimensions()
    {
        return _levelDimensions;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    tdns::common::K_DynamicArray3dDevice<uint32_t>& K_MultiResolutionPageDirectory<T>::get_level_coordinates()
    {
        return _levelCoordinates;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    uint32_t K_MultiResolutionPageDirectory<T>::get_nb_levels_of_resolution()
    {
        return _levelDimensions.size();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    uint3 K_MultiResolutionPageDirectory<T>::get_coords(uint32_t level, const float3 &position) const
    {
        const uint32_t &levelCoordinates = _levelCoordinates[level];
        const uint3 &levelDimensions = _levelDimensions[level];

        uint3 texCoord;

        texCoord.x = levelCoordinates + static_cast<uint32_t>(position.x * levelDimensions.x);
        texCoord.y = static_cast<uint32_t>(position.y * levelDimensions.y);
        texCoord.z = static_cast<uint32_t>(position.z * levelDimensions.z);

        return texCoord;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    T K_MultiResolutionPageDirectory<T>::get(const uint3 &texCoord) const
    {
        return _data.get(texCoord);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    T K_MultiResolutionPageDirectory<T>::get(uint32_t level, const float3 &position) const
    {
        const uint32_t &levelCoordinates = _levelCoordinates[level];
        const uint3 &levelDimensions = _levelDimensions[level];

        uint3 texCoord;

        texCoord.x = levelCoordinates + static_cast<uint32_t>(position.x * levelDimensions.x);
        texCoord.y = static_cast<uint32_t>(position.y * levelDimensions.y);
        texCoord.z = static_cast<uint32_t>(position.z * levelDimensions.z);

        return _data.get(texCoord);
    }
    
    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    void K_MultiResolutionPageDirectory<T>::set(const uint3 &position, const T &value)
    {
        _data.set(position, value);
    }
    
    //---------------------------------------------------------------------------------------------
    template<typename T>
    __device__
    void K_MultiResolutionPageDirectory<T>::set(uint32_t level, const float3 &position, const T &value) const
    {
        const uint32_t &levelCoordinates = _levelCoordinates[level];
        const uint3 &levelDimensions = _levelDimensions[level];

        uint3 surfCoord;
        
        surfCoord.x = levelCoordinates + static_cast<uint32_t>(position.x * levelDimensions.x);
        surfCoord.y = static_cast<uint32_t>(position.y * levelDimensions.y);
        surfCoord.z = static_cast<uint32_t>(position.z * levelDimensions.z);

        _data.set(surfCoord, value);
    }
} // namespace gpucache
} // namespace tdns