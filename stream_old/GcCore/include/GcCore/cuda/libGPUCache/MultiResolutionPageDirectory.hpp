#pragma once

#include <memory>
#include <vector>

#include <cuda.h>

#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/cuda/libCommon/Surface3dDevice.hpp>
#include <GcCore/cuda/libCommon/Texture3dDevice.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dDevice.hpp>
#include <GcCore/libCommon/Memory.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>

#include <GcCore/cuda/libGPUCache/K_MultiResolutionPageDirectory.hpp>

namespace tdns
{
namespace gpucache
{
    /**
     * @brief Class for a Multi resolution page directory (MRPD).
     * With t = 1 => 1 entry of the MRPD addresses 1 brick !
     * with t = 2 => 1 entry addresses 1 blocks of page table entry
     *               and 1 table entry addresses 1 brick.
     */
    template<typename T>
    class MultiResolutionPageDirectory : public 
        tdns::common::KernelObject<K_MultiResolutionPageDirectory<T>>,
        tdns::common::Noncopyable
    {
    public:

        /**
        * @brief Constructor.
        *
        * @param levelDimensions    Number of elements to store in the surface.
        * @param elementSize        Size of one element.
        *
        *      e.g. elementSize = (32, 32, 32) = element addressed per entry
        */
        MultiResolutionPageDirectory(
            const std::vector<uint3> &levelDimensions);

        /**
        * @brief Destructor.
        */
        virtual ~MultiResolutionPageDirectory() = default;

        /**
        * @brief Getters / Setters
        * 
        * @return The data.
        */
        tdns::common::Surface3dDevice<T>* data();

        /**
        * @brief Getters / Setters
        * 
        * @return The data.
        */
        const tdns::common::Surface3dDevice<T>* data() const;

        /**
        * 
        */
        virtual K_MultiResolutionPageDirectory<T> to_kernel_object() override;

        /**
        * @brief
        */
        tdns::common::DynamicArray3dDevice<uint3>& get_level_dimensions();
        const tdns::common::DynamicArray3dDevice<uint3>& get_level_dimensions() const;

    protected:
        /**
        * Member data.
        */
        tdns::common::DynamicArray3dDevice<uint32_t>        _levelCoordinates;  ///< 3D positions of the first entry of each level of resolution.
        tdns::common::DynamicArray3dDevice<uint3>           _levelDimensions;   ///< 3D sizes of each level of resolution.
        std::unique_ptr<tdns::common::Surface3dDevice<T>>   _data;              ///< Addresses in a surface.
        std::unique_ptr<tdns::common::Texture3dDevice<T>>   _texture;           ///< Addresses in a texture.

    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline MultiResolutionPageDirectory<T>::MultiResolutionPageDirectory(
        const std::vector<uint3> &levelDimensions)
        :   _levelCoordinates(tdns::math::Vector3ui(levelDimensions.size(), 1, 1)),
            _levelDimensions(tdns::math::Vector3ui(levelDimensions.size(), 1, 1))
    {
        LOGINFO(40, tdns::common::log_details::Verbosity::INSANE, "Creating MultiResolutionPageDirectory.");
#if TDNS_OS != TDNS_OS_WINDOWS //erreur de compile sur windows... il faut regarder ca...
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " `- levelDimensions = [" << levelDimensions.size() << "]");
#endif

        _levelDimensions  = levelDimensions;

        // Calculate the size of the MRPD - create the surface
        tdns::math::Vector3ui size(0);

        for (auto it = levelDimensions.begin(); it != levelDimensions.end(); ++it)
        {
            size[0] += it->x;
            size[1] = std::max(size[1], it->y);
            size[2] = std::max(size[2], it->z);
        }

        _data = tdns::common::create_unique_ptr<tdns::common::Surface3dDevice<T>>(size, tdns::math::Vector3ui(1));
        _texture = tdns::common::create_unique_ptr<tdns::common::Texture3dDevice<T>>(*_data, size, tdns::math::Vector3ui(1),
            tdns::common::Texture3dDevice<T>::AccessMode::Default);

        // Calculate the x offset of the begining of each level in the surface (not necessary on y and z)
        std::vector<uint32_t> levelCoordinates(levelDimensions.size());

        levelCoordinates[0] = 0;
        for (uint32_t i = 1; i < levelDimensions.size(); ++i)
        {
            levelCoordinates[i] = levelCoordinates[i - 1] + levelDimensions[i - 1].x;
        }

        _levelCoordinates = levelCoordinates;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline tdns::common::Surface3dDevice<T>* MultiResolutionPageDirectory<T>::data()
    {
        return _data.get();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::common::Surface3dDevice<T>* MultiResolutionPageDirectory<T>::data() const
    {
        return _data.get();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline K_MultiResolutionPageDirectory<T> MultiResolutionPageDirectory<T>::to_kernel_object()
    {
        return K_MultiResolutionPageDirectory<T>(*_data, *_texture, _levelCoordinates, _levelDimensions);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline tdns::common::DynamicArray3dDevice<uint3>& MultiResolutionPageDirectory<T>::get_level_dimensions()
    {
        return _levelDimensions;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::common::DynamicArray3dDevice<uint3>& MultiResolutionPageDirectory<T>::get_level_dimensions() const
    {
        return _levelDimensions;
    }
} // namespace gpucache
} // namespace tdns