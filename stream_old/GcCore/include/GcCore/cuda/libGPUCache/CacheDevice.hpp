#pragma once

#include <memory>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/libCommon/Memory.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/Texture3dDevice.hpp>

#include <GcCore/libMath/Vector.hpp>

#include <GcCore/cuda/libGPUCache/K_CacheDevice.hpp>
#include <GcCore/cuda/libGPUCache/LruContent.hpp>
#include <GcCore/cuda/libGPUCache/Predicate.hpp>

namespace tdns
{
namespace gpucache
{
    enum CacheDeviceFlag
    {
        NormalizedAccess = 1
    };
    /**
    * @brief
    */
    template<typename T>
    class CacheDevice : public tdns::common::KernelObject<K_CacheDevice<T>>, tdns::common::Noncopyable
    {
    public:

        /**
        * @brief Constructor.
        * 
        * @param cacheSize                  Number of elements to store in the surface.
        * @param blockSize                  Size of one element.
        * @param virtualizedNumberOfEntries Number of entries virtualized by the parent.
        * @param realNumberOfEntries        Real number of entries for the virtualize level.
        * @param timestamp                  Pointer to the timestamp.
        *
        *     e.g.  cacheSize = (10, 1, 1) bricks.
        *           blockSize = (32, 32, 32) = the size of a brick.
        *           virtualizedNumberOfEntries = (20, 20, 15) size of the whole virtual volume.
        *           realNumberOfEntries        = (15, 20, 10) size of the whole volume.
        */
        CacheDevice(const tdns::math::Vector3ui &cacheSize,
                    const tdns::math::Vector3ui &blockSize,
                    const std::vector<uint3> &virtualizedNumberOfEntries,
                    const std::vector<uint3> &realNumberOfEntries,
                    uint32_t *timestamp,
                    uint32_t flag = 0);

        /**
        * @brief Get an object that can be send to a CUDA kernel.
        *
        * return A CacheDevice object that can be send to a CUDA kernel.
        */
        virtual K_CacheDevice<T> to_kernel_object() override;

        /**
        * @brief
        * 
        * @param 
        * @param Values used as reference tested between mask and current data
        */
        template<typename U>
        void update(const thrust::device_vector<U> &masks, const U &ref);

        /**
         * @brief 
         */
        thrust::device_vector<LruContent>& get_lru();

        /**
         * @brief 
         */
        tdns::common::DynamicArray3dDevice<uint32_t>& get_usage_buffer();

        /**
         * @brief
         */
        const tdns::math::Vector3ui& get_block_size() const;

        /**
        * @brief
        */
        tdns::common::DynamicArray3dDevice<uint3>& get_real_number_of_entries();
        const tdns::common::DynamicArray3dDevice<uint3>& get_real_number_of_entries() const;

        /**
        * @brief
        */
        tdns::common::DynamicArray3dDevice<uint3>& get_virtual_number_of_entries();
        const tdns::common::DynamicArray3dDevice<uint3>& get_virtual_number_of_entries() const;

    protected:
        /**
        * Member data.
        */
        std::unique_ptr<tdns::common::Surface3dDevice<T>>   _data;                          ///<
        std::unique_ptr<tdns::common::Texture3dDevice<T>>   _texture;                       ///<
        thrust::device_vector<LruContent>                   _lru;                           ///< Contains the 3D positions in the surface.
        tdns::common::DynamicArray3dDevice<uint3>           _virtualizedNumberOfEntries;    ///< Size of the virtualized volume / block per level
        tdns::common::DynamicArray3dDevice<uint3>           _realNumberOfEntries;           ///< Size of the real volume / block per level
        tdns::common::DynamicArray3dDevice<float3>          _realDivVirtualNumberOfEntries; ///< Size of the real volume / block per level
        tdns::math::Vector3ui                               _blockSize;                     ///< Size of each blocks. e.g. if volume => brick size.
        tdns::common::DynamicArray3dDevice<uint32_t>        _usage;                         ///<
        const uint32_t                                      *_timestamp;                    ///<
    
    private:
        tdns::common::DynamicArray3dDevice<uint32_t>        _levelCoordinates;              ///< offset on the x axis of the first entry of each level of resolution.
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline CacheDevice<T>::CacheDevice(
        const tdns::math::Vector3ui &cacheSize,
        const tdns::math::Vector3ui &blockSize,
        const std::vector<uint3> &virtualizedNumberOfEntries,
        const std::vector<uint3> &realNumberOfEntries,
        uint32_t *timestamp,
        uint32_t flag /* = 0*/)
        :   _virtualizedNumberOfEntries(tdns::math::Vector3ui(virtualizedNumberOfEntries.size(), 1, 1)),
            _realNumberOfEntries(tdns::math::Vector3ui(realNumberOfEntries.size(), 1, 1)),
            _realDivVirtualNumberOfEntries(tdns::math::Vector3ui(realNumberOfEntries.size(), 1, 1)),
            _usage(cacheSize, 0),
            _timestamp(timestamp),
            _levelCoordinates(tdns::math::Vector3ui(realNumberOfEntries.size(), 1, 1))
    {
        LOGINFO(40, tdns::common::log_details::Verbosity::INSANE, "Creating CacheDevice.");
#if TDNS_OS != TDNS_OS_WINDOWS //erreur de compile sur windows... il faut regarder ca...
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " |- cacheSize = ("
            << cacheSize[0] << ", " << cacheSize[1] << ", " << cacheSize[2] << ")");
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " |- blockSize = ("
            << blockSize[0] << ", " << blockSize[1] << ", " << blockSize[2] << ")");
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " |- virtualizedNumberOfEntries = [" << virtualizedNumberOfEntries.size() << "]");
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " `- realNumberOfEntries = [" << realNumberOfEntries.size() << "]");
#endif
        _data = tdns::common::create_unique_ptr<tdns::common::Surface3dDevice<T>>(cacheSize, blockSize);
        _texture = tdns::common::create_unique_ptr<tdns::common::Texture3dDevice<T>>(*_data, cacheSize, blockSize,
            flag & CacheDeviceFlag::NormalizedAccess);
        _lru.resize(cacheSize[0] * cacheSize[1] * cacheSize[2]);

        //init the LRU
        uint32_t xy = cacheSize[1] * cacheSize[0];
        for (uint32_t z = 0; z < cacheSize[2]; ++z)
            for (uint32_t y = 0; y < cacheSize[1]; ++y)
                for (uint32_t x = 0; x < cacheSize[0]; ++x)
                {
                    LruContent tmp;
                    tmp.cachePosition = make_uint3(x * blockSize[0], y * blockSize[1], z * blockSize[2]);
                    tmp.volumePosition = make_float3(-1.f, -1.f, -1.f);
                    tmp.level = 0;
                    _lru[x + y * cacheSize[1] + z * xy] = tmp;

                    // _lru[x + y * cacheSize[1] + z * xy].cachePosition = 
                    //     make_uint3(x * blockSize[0], y * blockSize[1], z * blockSize[2]);
                }

        _blockSize = blockSize;
        _virtualizedNumberOfEntries = virtualizedNumberOfEntries;
        _realNumberOfEntries        = realNumberOfEntries;

        for (size_t i = 0; i < realNumberOfEntries.size(); ++i)
            *_realDivVirtualNumberOfEntries[i] = make_float3(
                static_cast<float>(realNumberOfEntries[i].x) / static_cast<float>(virtualizedNumberOfEntries[i].x),
                static_cast<float>(realNumberOfEntries[i].y) / static_cast<float>(virtualizedNumberOfEntries[i].y),
                static_cast<float>(realNumberOfEntries[i].z) / static_cast<float>(virtualizedNumberOfEntries[i].z)
            );

        *_levelCoordinates[0] = 0;
        for(size_t i = 1; i < _levelCoordinates.size(); ++i)
            *_levelCoordinates[i] = *_levelCoordinates[i -1] + realNumberOfEntries[i - 1].x;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline K_CacheDevice<T> CacheDevice<T>::to_kernel_object()
    {
        return K_CacheDevice<T>(
            *_data,
            *_texture,
            _virtualizedNumberOfEntries,
            _realNumberOfEntries,
            _realDivVirtualNumberOfEntries,
            _blockSize,
            _usage,
            *_timestamp,
            _levelCoordinates);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    template<typename U>
    inline void CacheDevice<T>::update(const thrust::device_vector<U> &masks, const U &ref)
    {
        thrust::device_vector<LruContent> result(_lru.size());
        
        // stream compaction to keep the used elements
        thrust::device_vector<LruContent>::iterator it = thrust::copy_if(
            _lru.begin(),
            _lru.end(),
            masks.begin(),
            result.begin(),
            predicate::equal<U>(ref));

        // stream compaction to keep the unused elements
        thrust::copy_if(_lru.begin(), _lru.end(), masks.begin(), it, predicate::not_equal<U>(ref));

        // update the LRU with the most recently used elements at the begining
        _lru = result;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline thrust::device_vector<LruContent>& CacheDevice<T>::get_lru()
    {
        return _lru;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline tdns::common::DynamicArray3dDevice<uint32_t>& CacheDevice<T>::get_usage_buffer()
    {
        return _usage;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::math::Vector3ui& CacheDevice<T>::get_block_size() const
    {
        return _blockSize;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline tdns::common::DynamicArray3dDevice<uint3>& CacheDevice<T>::get_real_number_of_entries()
    {
        return _realNumberOfEntries;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::common::DynamicArray3dDevice<uint3>& CacheDevice<T>::get_real_number_of_entries() const
    {
        return _realNumberOfEntries;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline tdns::common::DynamicArray3dDevice<uint3>& CacheDevice<T>::get_virtual_number_of_entries()
    {
        return _virtualizedNumberOfEntries;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::common::DynamicArray3dDevice<uint3>& CacheDevice<T>::get_virtual_number_of_entries() const
    {
        return _virtualizedNumberOfEntries;
    }
} // namespace gpucache
} // namespace tdns