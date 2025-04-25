#pragma once

#include <cstdint>
#include <cassert>
#include <climits>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/K_DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/DefaultType.hpp>

#include <GcCore/cuda/libGPUCache/K_MultiResolutionPageDirectory.hpp>
#include <GcCore/cuda/libGPUCache/K_CacheDevice.hpp>
#include <GcCore/cuda/libGPUCache/MultiResolutionPageDirectory.hpp>
#include <GcCore/cuda/libGPUCache/CacheDevice.hpp>

namespace tdns
{
namespace gpucache
{
    /**
    * @brief Enumeration to know if an asked voxel has been found or not.
    */
    enum VoxelStatus
    {
        Mapped = 0,
        Unmapped,
        Empty
    };

    /**
     * @brief
     */
    template<typename T>
    class K_CacheManager
    {
    public:
        /**
        * @brief Default constructor.
        */
        K_CacheManager( MultiResolutionPageDirectory<uint4> &mrpd,
                        CacheDevice<T> &cache,
                        K_CacheDevice<uint4> *tableCaches,
                        size_t nbTableCaches,
                        tdns::common::DynamicArray3dDevice<uint32_t> &requestBuffer,
                        tdns::common::DynamicArray3dDevice<uint8_t> &dataCacheMask,
                        tdns::common::DynamicArray3dDevice<float3> &initialOverRealSize,
                        uint32_t timeStamp,
                        uint32_t brickSize,
                        uint32_t covering);

        /**
        * @brief
        * 
        * @param  position [description]
        * @param  level    [description]
        * @param[out] output    Voxel's value.
        *
        * @return          [description]
        */
        __device__ VoxelStatus get(uint32_t level, const float3 &position, T &output);
        
        /**
        * @brief
        * 
        * @param  position [description]
        * @param  level    [description]
        * @return          [description]
        */
        template<typename U>
        __device__ VoxelStatus get_normalized(uint32_t level, const float3 &position, U &output);

        /**
        * @ brief
        */
        __device__ K_CacheDevice<T>& get_data_cache();
        __device__ const K_CacheDevice<T>& get_data_cache() const;

        /**
        * @ brief
        */
        __device__ K_MultiResolutionPageDirectory<uint4>& get_mrpd();
        __device__ const K_MultiResolutionPageDirectory<uint4>& get_mrpd() const;

        /**
        * @ brief
        */
        __device__ K_CacheDevice<uint4>* get_table_caches();
        __device__ const K_CacheDevice<uint4>* get_table_caches() const;

        /**
        * @ brieftdns::common::DynamicArray3dDevice<uint8_t> &dataCacheMask,
        */
        __device__ uint32_t get_nb_table_caches() const;

        /**
        * @ brief
        */
        __device__ void reset_data_cache_buffer_entries(uint32_t level, const float3 &position);

        __device__ uint8_t get_data_cache_buffer_entry(uint32_t level, const float3 &position);

        __device__ void set_data_cache_buffer_entry(uint32_t level, const float3 &position, const uint8_t value);

        __device__ uint3 compute_element_position(uint32_t level, const float3 &position);

    private:

        /**
        * @brief
        * 
        * @param level    [description]
        * @param position [description]
        * @param cam_dist
        */
        __device__ void raise_brick_request(uint32_t level, const float3 &position);

    public:
        /*
        * Member data
        */
        uint32_t                                        _timeStamp;             ///<
        uint32_t                                        _covering;              ///<
        uint32_t                                        _brickSize;             ///<
        tdns::common::K_DynamicArray3dDevice<uint32_t>  _requestBuffer;         ///<
        tdns::common::K_DynamicArray3dDevice<uint8_t>   _dataCacheMask;         ///<
        tdns::common::K_DynamicArray3dDevice<float3>    _initialOverRealSize;   ///<
        K_CacheDevice<uint4>                            *_tableCaches;          ///<
        K_CacheDevice<T>                                _dataCache;             ///<
        K_MultiResolutionPageDirectory<uint4>           _mrpd;                  ///<
        size_t                                          _nbTableCaches;         ///<
        uint3                                           _previousCoords;        ///<
        uint4                                           _previousEntries;       ///<
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline K_CacheManager<T>::K_CacheManager(MultiResolutionPageDirectory<uint4> &mrpd,
                                    CacheDevice<T> &dataCache,
                                    K_CacheDevice<uint4> *tableCaches,
                                    size_t nbTableCaches,
                                    tdns::common::DynamicArray3dDevice<uint32_t> &requestBuffer,
                                    tdns::common::DynamicArray3dDevice<uint8_t> &dataCacheMask,
                                    tdns::common::DynamicArray3dDevice<float3> &initialOverRealSize,
                                    uint32_t timeStamp,
                                    uint32_t brickSize,
                                    uint32_t covering)
    : _mrpd(mrpd.to_kernel_object()), _dataCache(dataCache.to_kernel_object())
    {
        _requestBuffer  = requestBuffer.to_kernel_object();
        _dataCacheMask  = dataCacheMask.to_kernel_object();
        _initialOverRealSize = initialOverRealSize.to_kernel_object();
        _tableCaches    = tableCaches;
        _timeStamp      = timeStamp;
        _brickSize       = brickSize;
        _covering       = covering;
        _nbTableCaches  = nbTableCaches;

        _previousCoords = make_uint3(UINT_MAX, UINT_MAX, UINT_MAX);
        _previousEntries = make_uint4(0, 0, 0, 0);
    }
    
    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ VoxelStatus K_CacheManager<T>::get(uint32_t level, const float3 &position, T &output)
    {
        uint32_t levelMax = _mrpd.get_nb_levels_of_resolution() - 1;
        uint32_t levelCurrent = level;

        uint4 entry;


        while(true)
        {
            bool checkOtherLevel = false;

            float3 realDivVirtual = _dataCache.get_real_div_virtual_number_of_entries()[level]; // 60 cycles
            const float3 &initialOverReal = _initialOverRealSize[level];

            float3 position_in_real_volume = make_float3
            (
                position.x * initialOverReal.x,
                position.y * initialOverReal.y,
                position.z * initialOverReal.z
            );

            float3 virtualPosition = make_float3
            (
                position_in_real_volume.x * realDivVirtual.x,
                position_in_real_volume.y * realDivVirtual.y,
                position_in_real_volume.z * realDivVirtual.z
            );

            entry = _mrpd.get(level, virtualPosition);

            // uint3 mrpdCoord = _mrpd.get_coords(level, virtualPosition);
            // if (mrpdCoord.x != _previousCoords.x || mrpdCoord.y != _previousCoords.y || mrpdCoord.z != _previousCoords.z)
            // {
            //     entry = _mrpd.get(mrpdCoord);
            //     _previousEntries = entry;
            //     _previousCoords = mrpdCoord;
            // }
            // else entry = _previousEntries;

            switch (entry.w)
            {
                case 1: // MAPPED
                {
                    for (uint32_t i = 0; i < _nbTableCaches; ++i)
                    {
                        entry = _tableCaches[i].get(entry, level, virtualPosition);
                        switch (entry.w)
                        {
                            case 0: // UNMAPPED
                            {
                                if (level == levelCurrent) raise_brick_request(level, position_in_real_volume);
                                
                                // << check other levels
                                if (level < levelMax)
                                {
                                    ++level;
                                    checkOtherLevel = true;
                                    break;
                                }
                                else //unmapped
                                {
                                    output = tdns::common::create_default<T>();
                                    return VoxelStatus::Unmapped;
                                }
                            }
                            case 2: // EMPTY
                                return VoxelStatus::Empty;
                            
                            // MAPPED : go to the next page table cache level
                        }
                    }
                    if (!checkOtherLevel)
                    {
                        output = _dataCache.get(entry, level, virtualPosition, _covering);
                        return VoxelStatus::Mapped;
                    }
                    break;
                }
                case 0: // UNMAPPED
                {
                    // raise brick request only for the asking level
                    if (level == levelCurrent) raise_brick_request(level, position_in_real_volume);
                    
                    // << check other levels
                    if (level < levelMax) ++level;
                    else
                    {
                        output = tdns::common::create_default<T>();
                        return VoxelStatus::Unmapped;
                    }
                    break;
                }
                case 2: // EMPTY
                {
                    return VoxelStatus::Empty;
                }
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    template<typename U>
    inline __device__ VoxelStatus K_CacheManager<T>::get_normalized(uint32_t level, const float3 &position, U &output)
    {
        uint32_t levelMax = _mrpd.get_nb_levels_of_resolution() - 1;
        uint32_t levelCurrent = level;
        
        uint4 entry;
        
        while(true)
        {
            bool checkOtherLevel = false;
            
            float3 realDivVirtual = _dataCache.get_real_div_virtual_number_of_entries()[level]; // 60
            const float3 &initialOverReal = _initialOverRealSize[level];

            float3 position_in_real_volume = make_float3
            (
                position.x * initialOverReal.x,
                position.y * initialOverReal.y,
                position.z * initialOverReal.z
            );

            // printf("Pos: %f %f %f | IOR: %f %f %f\n ", position.x, position.y, position.z, initialOverReal.x, initialOverReal.y, initialOverReal.z);

            float3 virtualPosition = make_float3
            (
                position_in_real_volume.x * realDivVirtual.x,
                position_in_real_volume.y * realDivVirtual.y,
                position_in_real_volume.z * realDivVirtual.z
            ); // 200 - 600
            
            uint3 mrpdCoord = _mrpd.get_coords(level, virtualPosition); // 400 - 1000

            if (mrpdCoord.x != _previousCoords.x || mrpdCoord.y != _previousCoords.y || mrpdCoord.z != _previousCoords.z)
            {
                entry = _mrpd.get(mrpdCoord);
                _previousEntries = entry;
                _previousCoords = mrpdCoord;
            }
            else entry = _previousEntries;
            // 600 - 1000

            switch (entry.w)
            {
                case 1: // MAPPED
                {
                    for (uint32_t i = 0; i < _nbTableCaches; ++i)
                    {
                        entry = _tableCaches[i].get(entry, level, virtualPosition);
                        switch (entry.w)
                        {
                            case 0: // UNMAPPED
                            {
                                if (level == levelCurrent) raise_brick_request(level, position_in_real_volume);
                                
                                // << check other levels
                                if (level < levelMax)
                                {
                                    ++level;
                                    checkOtherLevel = true;
                                    break;
                                }
                                else
                                {
                                    output = tdns::common::create_default<U>();
                                    return VoxelStatus::Unmapped;
                                }
                            }
                            case 2: // EMPTY
                                return VoxelStatus::Empty;
                            
                            // MAPPED : go to the next page table cache level
                        }
                    }
                    if (!checkOtherLevel)
                    {
                        output = _dataCache.template get_normalized<U>(entry, level, virtualPosition, _covering); // 1500 - 6000 (2500)
                        return VoxelStatus::Mapped;
                    }
                    break;
                }
                case 0: // UNMAPPED
                {
                    // raise brick request only for the asking level
                    if (level == levelCurrent) raise_brick_request(level, position_in_real_volume); // 600
                    
                    // << check other levels
                    if (level < levelMax) ++level;
                    else
                    {
                        output = tdns::common::create_default<U>();
                        return VoxelStatus::Unmapped;
                    }
                    break;
                }
                case 2: // EMPTY
                {
                    return VoxelStatus::Empty;
                }
            }
        }
    }

    //---------------------------------------------------------------------------------------------------

    template<typename T>
    inline __device__ void K_CacheManager<T>::raise_brick_request(uint32_t level, const float3 &position)
    {
        uint3 brickPosition;
        tdns::common::K_DynamicArray3dDevice<uint3> levelDimensions;
        tdns::common::K_DynamicArray3dDevice<uint32_t> levelCoordinates;

        uint32_t coordinates;
        uint3 dimensions;

        // if we get multiple level in the page table hierarchy
        if (_nbTableCaches != 0)
        {
            levelDimensions  = _tableCaches[_nbTableCaches - 1].get_real_number_of_entries();
            levelCoordinates = _tableCaches[_nbTableCaches - 1].get_level_coordinates();
        }
        else // only the MRPD
        {
            levelDimensions  = _mrpd.get_level_dimensions();
            levelCoordinates = _mrpd.get_level_coordinates();
        }

        // We just need the x dimensions because all the levels are summed on x and y, z are the size max
        dimensions  = levelDimensions(make_uint3(level, 0, 0));
        // offset on x, according to the level
        coordinates = levelCoordinates(make_uint3(level, 0, 0));

        brickPosition.x = coordinates + std::min(static_cast<uint32_t>(position.x * dimensions.x), dimensions.x - 1);
        brickPosition.y = std::min(static_cast<uint32_t>(position.y * dimensions.y), dimensions.y - 1);
        brickPosition.z = std::min(static_cast<uint32_t>(position.z * dimensions.z), dimensions.z - 1);        
        // if (brickPosition.x >= (dimensions.x + coordinates) || brickPosition.y >= dimensions.y || brickPosition.z >= dimensions.z) {
        //     printf("brickPosition: %d %d %d (%d %d %d)\n", 
        //         brickPosition.x, brickPosition.y, brickPosition.z,
        //         dimensions.x, dimensions.y, dimensions.z
        //     );
        // }

        if (_requestBuffer(brickPosition) <= _timeStamp)
            _requestBuffer(brickPosition) = _timeStamp + 1;
        else if (_requestBuffer(brickPosition) < _timeStamp + 1000)
            _requestBuffer(brickPosition)++;

    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ void K_CacheManager<T>::reset_data_cache_buffer_entries(uint32_t level, const float3 &position)
    {
        uint3 brickPosition = compute_element_position(level, position);
        uint3 brickDims = _dataCacheMask.dims();

        for(uint32_t x = 0; x < _brickSize; ++x)
        for(uint32_t y = 0; y < _brickSize; ++y)
        for(uint32_t z = 0; z < _brickSize; ++z)
        {
            uint3 position = make_uint3(
                std::min(brickPosition.x + x, brickDims.x-1), 
                std::min(brickPosition.y + y, brickDims.y-1), 
                std::min(brickPosition.z + z, brickDims.z-1)
            );
            _dataCacheMask(position) = 0;
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ uint8_t K_CacheManager<T>::get_data_cache_buffer_entry(uint32_t level, const float3 &position)
    {
        uint3 elementPosition = compute_element_position(level, position);
        
        return _dataCacheMask(elementPosition);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ void K_CacheManager<T>::set_data_cache_buffer_entry(uint32_t level, const float3 &position, const uint8_t value)
    {
        uint3 elementPosition = compute_element_position(level, position);

        _dataCacheMask(elementPosition) = value;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ uint3 K_CacheManager<T>::compute_element_position(uint32_t level, const float3 &position)
    {
        uint4 entry;
        
        float3 realDivVirtual = _dataCache.get_real_div_virtual_number_of_entries()[level];
        const float3 &initialOverReal = _initialOverRealSize[level];

        float3 virtualPosition = make_float3
        (
            position.x * realDivVirtual.x * initialOverReal.x,
            position.y * realDivVirtual.y * initialOverReal.y,
            position.z * realDivVirtual.z * initialOverReal.z
        );
        
        entry = _mrpd.get(level, virtualPosition);

        for (uint32_t i = 0; i < _nbTableCaches; ++i)
            entry = _tableCaches[i].get(entry, level, virtualPosition);

        uint3 virtualizedNumberOfEntries = _dataCache.get_virtualized_number_of_entries()[level];
        
        uint3 elementPosition;

        elementPosition.x = entry.x + static_cast<uint32_t>((position.x * virtualizedNumberOfEntries.x)) % _brickSize;
        elementPosition.y = entry.y + static_cast<uint32_t>((position.y * virtualizedNumberOfEntries.y)) % _brickSize;
        elementPosition.z = entry.z + static_cast<uint32_t>((position.z * virtualizedNumberOfEntries.z)) % _brickSize;
        
        return elementPosition;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ K_CacheDevice<T>& K_CacheManager<T>::get_data_cache()
    {
        return _dataCache;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ const K_CacheDevice<T>& K_CacheManager<T>::get_data_cache() const
    {
        return _dataCache;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ K_MultiResolutionPageDirectory<uint4>& K_CacheManager<T>::get_mrpd()
    {
        return _mrpd;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ const K_MultiResolutionPageDirectory<uint4>& K_CacheManager<T>::get_mrpd() const
    {
        return _mrpd;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ K_CacheDevice<uint4>* K_CacheManager<T>::get_table_caches()
    {
        return _tableCaches;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ const K_CacheDevice<uint4>* K_CacheManager<T>::get_table_caches() const
    {
        return _tableCaches;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__ uint32_t K_CacheManager<T>::get_nb_table_caches() const
    {
        return _nbTableCaches;
    }
} // namespace gpucache
} // namespace tdns