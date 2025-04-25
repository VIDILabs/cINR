#include <GcCore/libPython/Py_Bricker.hpp>
#include <GcCore/libPreprocessor/Bricker_v2.hpp>
#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/ConfigurationPolicies.hpp>
#include <GcCore/libData/BricksManager.hpp>

//---------------------------------------------------------------------------------------------------
bool process_bricking(const Py_BrickingConfiguration& conf)
{
    tdns::preprocessor::BrickingConfiguration cpp_conf;

    cpp_conf.volumeDirectory = conf.volumeDirectory;
    cpp_conf.volumeFileName = std::string(conf.volumeFileName);
    cpp_conf.outputDirectory = std::string(conf.outputDirectory);
    cpp_conf.level = conf.level;
    cpp_conf.startX = conf.startX;
    cpp_conf.startY = conf.startY;
    cpp_conf.startZ = conf.startZ;
    cpp_conf.endX = conf.endX;
    cpp_conf.endY = conf.endY;
    cpp_conf.endZ = conf.endZ;
    cpp_conf.levelDimensionX = conf.levelDimensionX;
    cpp_conf.levelDimensionY = conf.levelDimensionY;
    cpp_conf.levelDimensionZ = conf.levelDimensionZ;
    cpp_conf.brickSize = tdns::math::Vector3ui(conf.brickSizeX, conf.brickSizeY, conf.brickSizeZ);
    cpp_conf.bigBrickSize = tdns::math::Vector3ui(conf.bigBrickSizeX, conf.bigBrickSizeY, conf.bigBrickSizeZ);
    cpp_conf.covering = conf.coveringX;
    cpp_conf.encodedBytes = conf.encodedBytes;
    cpp_conf.compression = conf.compression;

    return tdns::preprocessor::process_bricking(cpp_conf);
}
