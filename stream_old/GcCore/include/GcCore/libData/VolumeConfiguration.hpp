#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libData/BrickKey.hpp>

namespace tdns
{
namespace data
{
    enum DataTypeEnum {
        TYPE_INVALID = 0,
        TYPE_UCHAR = 8000,
        TYPE_USHORT,
        TYPE_UINT,
        TYPE_CHAR,
        TYPE_SHORT,
        TYPE_INT,
        TYPE_FLOAT,
        TYPE_DOUBLE,
    };

    inline DataTypeEnum getDataTypeEnum(std::string name) { 
        if (name == "uchar")  return TYPE_UCHAR;
        if (name == "ushort") return TYPE_USHORT;
        if (name == "uint")   return TYPE_UINT;
        if (name == "uint8")  return TYPE_UCHAR;
        if (name == "uint16") return TYPE_USHORT;
        if (name == "uint32") return TYPE_UINT;
        if (name == "char")   return TYPE_CHAR;
        if (name == "short")  return TYPE_SHORT;
        if (name == "int")    return TYPE_INT;
        if (name == "int8")  return TYPE_CHAR;
        if (name == "int16") return TYPE_SHORT;
        if (name == "int32") return TYPE_INT;
        if (name == "float") return TYPE_FLOAT;
        if (name == "double") return TYPE_DOUBLE;
        return TYPE_INVALID;
    }

    inline uint32_t getDataTypeSize(DataTypeEnum type) { 
        if (type == TYPE_UCHAR)  return sizeof(uint8_t);
        if (type == TYPE_USHORT) return sizeof(uint16_t);
        if (type == TYPE_UINT)   return sizeof(uint32_t);
        if (type == TYPE_CHAR)   return sizeof(int8_t);
        if (type == TYPE_SHORT)  return sizeof(int16_t);
        if (type == TYPE_INT)    return sizeof(int32_t);
        if (type == TYPE_FLOAT)  return sizeof(float);
        if (type == TYPE_DOUBLE) return sizeof(double);
        throw std::runtime_error("Invalid type: " + std::to_string(type));
    }

    struct TDNS_API VolumeConfiguration
    {
        std::string             VolumeDirectory;    ///< Path where the volume is.
        std::string             VolumeFileName;     ///< Name of the volume (with the extension).
        std::string             INR_Path;           ///< Path to model params
        tdns::math::Vector3ui   BrickSize;          ///< Size of a brick.
        tdns::math::Vector3ui   BigBrickSize;       ///< Number of bricks in a big brick.
        tdns::math::Vector3ui   Covering;           ///< Number of overlapping voxels.
        uint32_t                EncodedBytes;       ///< Number of byte a voxel is encoded on.
        uint32_t                Channels;           ///< Number of byte a voxel is encoded on.
        uint32_t                NbLevels;           ///< Number of levels for the volume.

        std::vector<tdns::math::Vector3ui>  InitialVolumeSizes;
        std::vector<tdns::math::Vector3ui>  RealVolumesSizes;
        std::vector<tdns::math::Vector3ui>  NbBricks;
        std::vector<tdns::math::Vector3ui>  NbBigBricks;
        std::vector<Bkey>                   EmptyBricks;
        std::vector<float>                  Histogram;
    };

    /**
    */
    VolumeConfiguration TDNS_API load_volume_configuration(const std::string &configurationFile);

    /**
    */
    void TDNS_API write_volume_configuration(const std::string &file);
} // namespace data
} // namespace tdns