#pragma once

#include <vector>
#include <cstdint>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/Surface3dDevice.hpp>
#include <GcCore/cuda/libCommon/K_Surface3dDevice.hpp>
#include <GcCore/cuda/libCommon/K_DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/K_Texture3dDevice.hpp>
#include <GcCore/cuda/libCommon/Texture3dDevice.hpp>

namespace tdns
{
namespace gpucache
{
    /**
    * 
    */
    template<typename T>
    class K_CacheDevice
    {
    public:

        /**
        * @brief Constructor.
        */
        K_CacheDevice(
            tdns::common::Surface3dDevice<T> &data,
            tdns::common::Texture3dDevice<T> &texture,
            tdns::common::DynamicArray3dDevice<uint3> &virtualizedNumberOfEntries,
            tdns::common::DynamicArray3dDevice<uint3> &realNumberOfEntries,
            tdns::common::DynamicArray3dDevice<float3> &realDivVirtualNumberOfEntries,
            const tdns::math::Vector3ui &blockSize,
            tdns::common::DynamicArray3dDevice<uint32_t> &usage,
            uint32_t timestamp,
            tdns::common::DynamicArray3dDevice<uint32_t> &levelCoordinates);

        /**
         * @brief
         * 
         * @param  tableEntry [description]
         * @param  position   [description]
         * @return            [description]
         */
        __device__ T get(const uint4 &tableEntry, uint32_t level, const float3 &position, uint32_t covering = 0);

        /**
         * @brief
         * 
         * @param  tableEntry [description]
         * @param  position   [description]
         * @return            [description]
         */
        template<typename U>
        __device__ U get_normalized(const uint4 &tableEntry, uint32_t level, const float3 &position, uint32_t covering = 0);

        /**
         * @brief
         * 
         * @param  position [description]
         * @return          [description]
         */
        __device__ T get(const uint3 &position);

        /**
        * @brief
        * 
        * @param position [description]
        * @param value    [description]
        */
        __device__ void insert(const uint3 &position, const T &value);
        __device__ void insert(const uint4 &tableEntry, uint32_t level, const float3 &position, const T &value);

        /**
         * @brief
         * 
         * @param position [description]
         */
        __device__ void update_usage_buffer(const uint3 &position);

        /**
         * @brief
         * 
         * @return [description]
         */
        __device__ tdns::common::K_DynamicArray3dDevice<uint3>& get_virtualized_number_of_entries();

        /**
         * @brief
         * 
         * @return [description]
         */
        __device__ tdns::common::K_DynamicArray3dDevice<uint3>& get_real_number_of_entries();
        
        /**
         * @brief
         * 
         * @return [description]
         */
        __device__ tdns::common::K_DynamicArray3dDevice<float3>& get_real_div_virtual_number_of_entries();
        

        /**
        *
        */
        __device__ tdns::common::K_DynamicArray3dDevice<uint32_t>& get_level_coordinates();

        /**
        * @brief
        */
        __device__ const uint3& get_block_size() const;

        /**
        * @brief
        *
        */
        __device__ void reset_entries(const uint3 &entryPosition);

    protected:

        __device__ uint3 compute_element_position(const uint4 &tableEntry, uint32_t level, const float3 &position, uint32_t covering);

    protected:
        /**
        * Member data.
        */
        uint32_t                                        _timestamp;                     ///<
        tdns::common::K_DynamicArray3dDevice<uint32_t>  _levelCoordinates;              ///<
        uint3                                           _blockSize;                     ///< Size of each blocks. e.g. if volume => brick size.
        tdns::common::K_Surface3dDevice<T>              _data;                          ///<
        tdns::common::K_Texture3dDevice<T>              _texture;                       ///<
        tdns::common::K_DynamicArray3dDevice<uint32_t>  _usage;                         ///<
        tdns::common::K_DynamicArray3dDevice<uint3>     _virtualizedNumberOfEntries;    ///< Size of the virtualized volume / block per level.
        tdns::common::K_DynamicArray3dDevice<uint3>     _realNumberOfEntries;           ///< Size of the real volume / block per level
        tdns::common::K_DynamicArray3dDevice<float3>   _realDivVirtualNumberOfEntries; ///<

    };
    
    //---------------------------------------------------------------------------------------------
    template<typename T>
    K_CacheDevice<T>::K_CacheDevice(
        tdns::common::Surface3dDevice<T> &data,
        tdns::common::Texture3dDevice<T> &texture,
        tdns::common::DynamicArray3dDevice<uint3> &virtualizedNumberOfEntries,
        tdns::common::DynamicArray3dDevice<uint3> &realNumberOfEntries,
        tdns::common::DynamicArray3dDevice<float3> &realDivVirtualNumberOfEntries,
        const tdns::math::Vector3ui &blockSize,
        tdns::common::DynamicArray3dDevice<uint32_t> &usage,
        uint32_t timestamp,
        tdns::common::DynamicArray3dDevice<uint32_t> &levelCoordinates)
        :   _data(data.to_kernel_object()),
            _texture(texture.to_kernel_object()),
            _usage(usage.to_kernel_object())
    {
        _virtualizedNumberOfEntries    = virtualizedNumberOfEntries.to_kernel_object();
        _realNumberOfEntries           = realNumberOfEntries.to_kernel_object();
        _realDivVirtualNumberOfEntries = realDivVirtualNumberOfEntries.to_kernel_object();

        _blockSize = *reinterpret_cast<const uint3*>(blockSize.data());

        _timestamp = timestamp;
        _levelCoordinates = levelCoordinates.to_kernel_object();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T K_CacheDevice<T>::get(const uint3 &position)
    {
        return _data.get(position);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T K_CacheDevice<T>::get(const uint4 &tableEntry, uint32_t level, const float3 &position, uint32_t covering /*= 0*/)
    {
        uint3 elementPosition = compute_element_position(tableEntry, level, position, covering);

        // elementPosition = position in the surface of the sample we are looking for
        return _texture.get(elementPosition);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    template<typename U>
    inline __device__
    U K_CacheDevice<T>::get_normalized(const uint4 &tableEntry, uint32_t level, const float3 &position, uint32_t covering /*= 0*/)
    {
        uint3 elementPosition = compute_element_position(tableEntry, level, position, covering);

        // NOTE: We are missing the fractional part of the position. The fractional part is 
        //       important for linear interpolation.
        uint3 virtualizedNumberOfEntries = _virtualizedNumberOfEntries[level];
        float3 elementCoord = make_float3(
            position.x * virtualizedNumberOfEntries.x,
            position.y * virtualizedNumberOfEntries.y,
            position.z * virtualizedNumberOfEntries.z);
        elementCoord.x = elementCoord.x - static_cast<uint32_t>(elementCoord.x) + elementPosition.x;
        elementCoord.y = elementCoord.y - static_cast<uint32_t>(elementCoord.y) + elementPosition.y;
        elementCoord.z = elementCoord.z - static_cast<uint32_t>(elementCoord.z) + elementPosition.z;

        float3 texturePosition = make_float3(
            (elementCoord.x / static_cast<float>(_texture.size().x)),
            (elementCoord.y / static_cast<float>(_texture.size().y)),
            (elementCoord.z / static_cast<float>(_texture.size().z)));

        return _texture.template get_normalized<U>(texturePosition);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    void K_CacheDevice<T>::insert(const uint3 &position, const T &value)
    {
        _data.set(position, value);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    void K_CacheDevice<T>::insert(const uint4 &tableEntry, uint32_t level, const float3 &position, const T &value)
    {
        uint3 virtualizedNumberOfEntries = _virtualizedNumberOfEntries[level];
        uint3 blockSize = _blockSize;

        uint3 elementPosition;

        elementPosition.x = tableEntry.x + static_cast<uint32_t>((position.x * virtualizedNumberOfEntries.x)) % blockSize.x;
        elementPosition.y = tableEntry.y + static_cast<uint32_t>((position.y * virtualizedNumberOfEntries.y)) % blockSize.y;
        elementPosition.z = tableEntry.z + static_cast<uint32_t>((position.z * virtualizedNumberOfEntries.z)) % blockSize.z;

        insert(elementPosition, value);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    void K_CacheDevice<T>::update_usage_buffer(const uint3 &position)
    {
        _usage(position) = _timestamp;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    tdns::common::K_DynamicArray3dDevice<uint3>& K_CacheDevice<T>::get_virtualized_number_of_entries()
    {
        return _virtualizedNumberOfEntries;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    tdns::common::K_DynamicArray3dDevice<uint3>& K_CacheDevice<T>::get_real_number_of_entries()
    {
        return _realNumberOfEntries;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    tdns::common::K_DynamicArray3dDevice<float3>& K_CacheDevice<T>::get_real_div_virtual_number_of_entries()
    {
        return _realDivVirtualNumberOfEntries;
    }

    //---------------------------------------------------------------------------------------------
     template<typename T>
    inline __device__
    tdns::common::K_DynamicArray3dDevice<uint32_t>& K_CacheDevice<T>::get_level_coordinates()
    {
        return _levelCoordinates;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const uint3& K_CacheDevice<T>::get_block_size() const
    {
        return _blockSize;
    }

    //---------------------------------------------------------------------------------------------
    template<>
    inline __device__
    void K_CacheDevice<uint4>::reset_entries(const uint3 &entryPosition)
    {
#if 0
//#if (__CUDA_ARCH__ >= 350)

#else
        for(uint32_t x = 0; x < _blockSize.x; ++x)
        for(uint32_t y = 0; y < _blockSize.y; ++y)
        for(uint32_t z = 0; z < _blockSize.z; ++z)
        {
            _data.set(make_uint3(entryPosition.x + x, entryPosition.y + y, entryPosition.z + z), make_uint4(0, 0, 0, 0));
        }
#endif

    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    void K_CacheDevice<T>::reset_entries(const uint3 &entryPosition)
    {
//#if (__CUDA_ARCH__ >= 350)
#if 0

#else
        for(uint32_t x = 0; x < _blockSize.x; ++x)
        for(uint32_t y = 0; y < _blockSize.y; ++y)
        for(uint32_t z = 0; z < _blockSize.z; ++z)
        {
            _data.set(make_uint3(entryPosition.x + x, entryPosition.y + y, entryPosition.z + z), T(0));
        }
#endif
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    uint3 K_CacheDevice<T>::compute_element_position(const uint4 &tableEntry, uint32_t level, const float3 &position, uint32_t covering)
    {
        uint3 virtualizedNumberOfEntries = _virtualizedNumberOfEntries[level];
        uint3 blockSize = _blockSize;

        uint3 elementPosition = make_uint3(tableEntry.x, tableEntry.y, tableEntry.z);
        // elementPosition = position in the surface of the begining of the element

        elementPosition.x /= blockSize.x;
        elementPosition.y /= blockSize.y;
        elementPosition.z /= blockSize.z;

        // elementPosition = position in the "usage buffer"
        update_usage_buffer(elementPosition);

        elementPosition.x = covering + tableEntry.x + static_cast<uint32_t>((position.x * virtualizedNumberOfEntries.x)) % (blockSize.x - 2 * covering);
        elementPosition.y = covering + tableEntry.y + static_cast<uint32_t>((position.y * virtualizedNumberOfEntries.y)) % (blockSize.y - 2 * covering);
        elementPosition.z = covering + tableEntry.z + static_cast<uint32_t>((position.z * virtualizedNumberOfEntries.z)) % (blockSize.z - 2 * covering);

        return elementPosition;
    }
} // namespace gpucache
} // namespace tdns