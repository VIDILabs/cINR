#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <mutex>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/LRUCache.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libData/Brick.hpp>
#include <GcCore/libData/BrickKey.hpp>
#include <GcCore/libData/MetaData.hpp>

#include <GcCore/libData/Configuration.hpp>
#include "../../../../../ovr/api.h"

namespace tdns
{
namespace data
{
    /**
    * @brief Singleton that manage the bricks in memory.
    * If the brick does not exist in RAM it will load it.
    */
    class TDNS_API BricksManager
    {
    public:
        /**
        * @brief Enum to give the status a of the requested brick.
        *
        */
        enum BrickStatus
        {
            Success = 0,    ///< Normal condition the brick has been found.
            Unknown,        ///< The requested brick is unknown.
            Empty           ///< The requested brick is empty (no data to load).
        };

    public:
        /**
        * @brief Default constructor.
        */
        BricksManager(const std::string &volumeDirectory, const tdns::math::Vector3ui &brickSize,
            const tdns::math::Vector3ui &bigBrickSize, uint32_t numberEncodedBytes, size_t cacheSize = 32768);

        /**
        * @brief Desctrutor.
        */
        virtual ~BricksManager() = default;
        
        /**
        * @brief Get a brick regarding its precision level and its position in the volume.
        *
        * @param Level precision of the brick.
        * @param Position of the brick in the volume
        *       The position is the brick number on the axis.
        *           eg. x = 1, y = 0, z = 2 means: the second brick on X axis, the first on Y axis,
        *               and the third on Z axis.
        * @return The brick if in cache or loaded from file, nullptr if not found.
        */
        Brick* get_brick(uint32_t level, const tdns::math::Vector3ui &position);

        /**
        * @brief Get a brick regarding its precision level and its position in the volume.
        *
        * @param Level precision of the brick.
        * @param Position of the brick in the volume
        *       The position is the brick number on the axis.
        *           eg. x = 1, y = 0, z = 2 means: the second brick on X axis, the first on Y axis,
        *               and the third on Z axis.
        * @param[out] The brick to return.
        *
        * @return The status of the brick.
        */
        BrickStatus get_brick(uint32_t level, const tdns::math::Vector3ui &position, Brick **brick, uint32_t i);
        void get_bricks(uint32_t* levels, tdns::math::Vector3ui* positions, Brick** bricks, BrickStatus* statusArr, const uint32_t count);

        /**
        * @brief Write a brick in a file.
        *
        * The brick file will be writen in "WorkingDirectory/bricks/level/brick.brick".
        *
        * @param The brick to write in a file.
        */
        static float write_brick(const std::string &outputDirectory,
            const Brick &brick,
            const tdns::math::Vector3ui &brickSize,
            bool compression);

        /**
        * @brief Load the empty bricks
        */
        void load(const MetaData &metaData);

        /**
        * @brief Get the string corresponding to the status
        */
        std::string get_status_string(BrickStatus status) const;

        static void check_level_directory(const std::string &volumeDirectory,
            uint32_t level,
            const tdns::math::Vector3ui &brickSize);

        /**
        * @brief Give the name of the folder path given the brick sizes.
        *
        * @param[in]    BrickSize   The brick size on all axes.
        *
        * @return The folder name.
        */
        static std::string get_brick_folder(const tdns::math::Vector3ui &brickSize);

    protected:

        /**
        * @brief Add the new brick in the map and LRU cache.
        *
        * Remove the oldest brick if the LRU cache has removed one.
        *
        * @param The new brick to add to the cache
        */
        void insert_in_cache(const Bkey &key, Brick *brick);

        /**
        * @brief Gives the full path to the brick.
        *
        * The path is "WorkingDirectory/bricks/LN/LN_X_Y_Z.brick" with,
        *       - N = level of precision (0 = max precision).
        *       - X = Brick numbore on X axis (0 means the first brick, 1 means the second brick, etc.).
        *       - Y = Brick numbore on Y axis (0 means the first brick, 1 means the second brick, etc.).
        *       - Z = Brick numbore on Z axis (0 means the first brick, 1 means the second brick, etc.).
        *   eg. ./bricks/L0/L0_1_8_2
        *
        * @param Base path to the folder containing the brick folder.
        * @param 3D Size of a brick.
        * @param Level of detail.
        * @param Position of the brick.
        */
        static std::string get_brick_path(const std::string &baseDirectory,
            const tdns::math::Vector3ui &brickSize,
            uint32_t level,
            const tdns::math::Vector3ui &position);

        /**
        * @brief Create the normalize folder name from the level.
        *
        * The folder name norm is: LN/ with
        *       - N = level of precion (0 = max precision).
        *   eg. L0/
        *
        * @param Level of precision.
        * @return The normalize folder name.
        */
        static std::string get_level_folder(uint32_t level);

        /**
        *
        *
        */
        Brick* load_brick(Bkey key, uint32_t level, const tdns::math::Vector3ui &position, const uint32_t i);
        void load_bricks(Bkey* keys, uint32_t* levels, tdns::math::Vector3ui* positions, Brick** bricks, const uint32_t count);

    public:
        /**
        * Member data
        */
        tdns::math::Vector3ui                       _brickEdgeSize;         ///< Edge size of a brick.
        tdns::math::Vector3ui                       _bigBrickSize;          ///< Number of bricks inside a big brick.
        uint32_t                                    _numberEncodedBytes;    ///< Number of bytes used to encode each voxel.
        std::string                                 _INRpath;               ///< Path to the model params.json

        std::map<Bkey, std::unique_ptr<Brick>>      _bricks;                ///< Cache of brick to store it when loaded from HD.
        std::set<Bkey>                              _emptyBricks;           ///< List of empty bricks.
        tdns::common::LRUCache<Bkey, Brick*>        _cache;                 ///< LRU cache to know which brick to release if not used.
        std::mutex                                  _lock;

        vnrNetwork                                  _net;                   ///< Injected model network
        vnr::vec3f                                  _volDims;               ///< Volume dims for coordinate normalization
        vnr::vec3f*                                 _d_coords;              ///< Coordinates to be offset for VNR on Device
        vnr::vec3f*                                 _d_outCoords;           ///< Coordinates for inference
        float*                                      _d_values;              ///< Output values from network
    };
} //namespace data
} //namespace tdns