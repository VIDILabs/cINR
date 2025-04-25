#pragma once

#include <string>
#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libData/MetaData.hpp>

namespace tdns
{
namespace preprocessor
{
    struct BrickingConfiguration
    {
        std::string volumeDirectory;        ///< Folder path where the volume is.
        std::string volumeFileName;         ///< Volume name with extension.
        std::string outputDirectory;        ///< Folder path where the brick files will be saved.
        uint32_t    level;                  ///< The level to brick.
        uint32_t    startX;                 ///< The voxel index in the volume on X-axis to begin the bricking.
        uint32_t    startY;                 ///< The voxel index in the volume on Y-axis to begin the bricking.
        uint32_t    startZ;                 ///< The voxel index in the volume on Z-axis to begin the bricking.
        uint32_t    endX;                   ///< The voxel index in the volume on X-axis to end the bricking.
        uint32_t    endY;                   ///< The voxel index in the volume on Y-axis to end the bricking.
        uint32_t    endZ;                   ///< The voxel index in the volume on Z-axis to end the bricking.
        uint32_t    levelDimensionX;        ///< Volume size on X-axis for the given level.
        uint32_t    levelDimensionY;        ///< Volume size on Y-axis for the given level.
        uint32_t    levelDimensionZ;        ///< Volume size on Z-axis for the given level.
        tdns::math::Vector3ui brickSize;    ///< Size of a brick on all axes.
        tdns::math::Vector3ui bigBrickSize; ///< Number of bricks in a big bricks on all axes.
        uint32_t    covering;               ///< Number of overlapping voxels.
        uint32_t    encodedBytes;           ///< Number of bytes a voxel is encoded.
        bool        compression;            ///< Save the bricks in compressed format or not.
    };

    void TDNS_API init_meta_data(tdns::data::MetaData &metaData, const BrickingConfiguration& conf,
        const std::vector<tdns::math::Vector3ui> &levels);

    /**
    * @brief
    */
    bool TDNS_API process_bricking(const BrickingConfiguration& conf);
} // namespace preprocessor
} // namespace tdns