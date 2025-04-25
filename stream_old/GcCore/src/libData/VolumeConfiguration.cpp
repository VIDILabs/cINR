#include <GcCore/libData/VolumeConfiguration.hpp>

#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libTinyXml/tinyxml2.h>
#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/ConfigurationPolicies.hpp>
#include <GcCore/libData/BricksManager.hpp>

namespace tdns
{
namespace data
{
    VolumeConfiguration TDNS_API load_volume_configuration(const std::string &configurationFile)
    {
        LOGTRACE(10, tdns::common::log_details::Verbosity::INSANE, "Load meta data from configuration [" << configurationFile << "].");

        tdns::data::Configuration &conf = Configuration::get_instance();
        conf.load<tdns::data::TDNSConfigurationParser>(configurationFile);

        VolumeConfiguration volumeConfiguration;

        // @TODO: Change this later to handle non cubic bricks
        uint32_t brickSize;
        if(!conf.get_field("BrickSize", brickSize))
            throw std::runtime_error("VolumeConfiguration: Missing BrickSize in configuration.");
        volumeConfiguration.BrickSize = tdns::math::Vector3ui(brickSize);

        if (!conf.get_field("BigBrickSizeX", volumeConfiguration.BigBrickSize[0]))
            throw std::runtime_error("VolumeConfiguration: Missing BigBrickSizeX in configuration.");
        if (!conf.get_field("BigBrickSizeY", volumeConfiguration.BigBrickSize[1]))
            throw std::runtime_error("VolumeConfiguration: Missing BigBrickSizeY in configuration.");
        if (!conf.get_field("BigBrickSizeZ", volumeConfiguration.BigBrickSize[2]))
            throw std::runtime_error("VolumeConfiguration: Missing BigBrickSizeZ in configuration.");

        uint32_t covering;
        if (!conf.get_field("VoxelCovering", covering))
            throw std::runtime_error("VolumeConfiguration: Missing VoxelCovering in configuration.");
        volumeConfiguration.Covering = tdns::math::Vector3ui(covering);

        if (!conf.get_field("NumberEncodedBytes", volumeConfiguration.EncodedBytes))
            throw std::runtime_error("VolumeConfiguration: Missing NumberEncodedBytes in configuration.");

        if (!conf.get_field("NumberChannels", volumeConfiguration.Channels))
            throw std::runtime_error("VolumeConfiguration: Missing NumberChannels in configuration.");


        if (!conf.get_field("VolumeFile", volumeConfiguration.VolumeFileName))
            throw std::runtime_error("VolumeConfiguration: Missing VolumeFile in configuration.");

        if (!conf.get_field("INR_Path", volumeConfiguration.INR_Path))
            throw std::runtime_error("VolumeConfiguration: Missing INR_Path in configuration.");

        //--------------------
        // @TODO: Need to fix this step
        std::string workingDirectory;
        conf.get_field("WorkingDirectory", workingDirectory);
        std::string volumeDirectory = workingDirectory + tdns::common::get_file_base_name(volumeConfiguration.VolumeFileName) + "/";
        conf.add_field("VolumeDirectory", volumeDirectory);
        //-----------------------------------------

        if (!conf.get_field("VolumeDirectory", volumeConfiguration.VolumeDirectory))
            throw std::runtime_error("VolumeConfiguration: Missing VolumeFile in configuration.");

        // std::string bricksDirectory = volumeConfiguration.VolumeDirectory + BricksManager::get_brick_folder(tdns::math::Vector3ui(brickSize));

        // std::string bricksFilePath = bricksDirectory + "/bricks.xml";

        // LOGTRACE(10, tdns::common::log_details::Verbosity::INSANE, "Loading Bricks.xml from [" << bricksFilePath << "].");

        // tinyxml2::XMLDocument doc;
        // tinyxml2::XMLError result = doc.LoadFile(bricksFilePath.c_str());

        // if (result != tinyxml2::XML_SUCCESS)
        //     throw std::runtime_error("VolumeConfiguration: Unable to open the file \"" +  bricksFilePath + "\". error [" + std::to_string(result) + "].");

        // tinyxml2::XMLNode *root = doc.FirstChild();
        // if (!root)
        //     throw std::runtime_error("VolumeConfiguration: No child in the file bricks.xml file. path [" + bricksFilePath + "].");

        // tinyxml2::XMLNode *levels = root->FirstChildElement("Levels");
        // if (!levels)
        //     throw std::runtime_error("VolumeConfiguration: No Levels meta data information in the file bricks.xml file. path [" + bricksFilePath + "].");

        // // Load intial and real levels sizes
        // tinyxml2::XMLElement *element = levels->FirstChildElement("Level");
        // while (element)
        // {
        //     uint32_t sizeX = std::atoi(element->Attribute("initial_size_X"));
        //     uint32_t sizeY = std::atoi(element->Attribute("initial_size_Y"));
        //     uint32_t sizeZ = std::atoi(element->Attribute("initial_size_Z"));
        //     volumeConfiguration.InitialVolumeSizes.push_back(tdns::math::Vector3ui(sizeX, sizeY, sizeZ));

        //     sizeX = std::atoi(element->Attribute("real_size_X"));
        //     sizeY = std::atoi(element->Attribute("real_size_Y"));
        //     sizeZ = std::atoi(element->Attribute("real_size_Z"));
        //     volumeConfiguration.RealVolumesSizes.push_back(tdns::math::Vector3ui(sizeX, sizeY, sizeZ));

        //     element = element->NextSiblingElement();
        // }

        uint32_t sizeX, sizeY, sizeZ;
        conf.get_field("size_X", sizeX); 
        conf.get_field("size_Y", sizeY);
        conf.get_field("size_Z", sizeZ);

        const uint32_t maxDim = std::max(sizeX, std::max(sizeY, sizeZ));
        const uint32_t levelMax = static_cast<uint32_t>(std::ceil(std::log(maxDim / brickSize) / (std::log(2)))+1);

        // Currently only cube bricks supported
        uint32_t brickSizeWithoutCovering = (brickSize - 2) * covering;
        for (uint32_t i = 0; i < levelMax; ++i) {
            uint32_t nbBricksX = static_cast<uint32_t>(std::ceil((float)sizeX / brickSizeWithoutCovering));
            uint32_t nbBricksY = static_cast<uint32_t>(std::ceil((float)sizeY / brickSizeWithoutCovering));
            uint32_t nbBricksZ = static_cast<uint32_t>(std::ceil((float)sizeZ / brickSizeWithoutCovering));

            nbBricksX = static_cast<uint32_t>(std::ceil((float)nbBricksX / (float)volumeConfiguration.BigBrickSize[0])) * volumeConfiguration.BigBrickSize[0];
            nbBricksY = static_cast<uint32_t>(std::ceil((float)nbBricksY / (float)volumeConfiguration.BigBrickSize[1])) * volumeConfiguration.BigBrickSize[1];
            nbBricksZ = static_cast<uint32_t>(std::ceil((float)nbBricksZ / (float)volumeConfiguration.BigBrickSize[2])) * volumeConfiguration.BigBrickSize[2];
            volumeConfiguration.NbBricks.push_back(tdns::math::Vector3ui(nbBricksX, nbBricksY, nbBricksZ));

            uint32_t nbBigBricksX = nbBricksX / volumeConfiguration.BigBrickSize[0];
            uint32_t nbBigBricksY = nbBricksY / volumeConfiguration.BigBrickSize[1];
            uint32_t nbBigBricksZ = nbBricksZ / volumeConfiguration.BigBrickSize[2];
            volumeConfiguration.NbBigBricks.push_back(tdns::math::Vector3ui(nbBigBricksX, nbBigBricksY, nbBigBricksZ));

            uint32_t realSizeX = nbBricksX * brickSizeWithoutCovering;
            uint32_t realSizeY = nbBricksY * brickSizeWithoutCovering;
            uint32_t realSizeZ = nbBricksZ * brickSizeWithoutCovering;

            volumeConfiguration.InitialVolumeSizes.push_back(tdns::math::Vector3ui(sizeX, sizeY, sizeZ));
            volumeConfiguration.RealVolumesSizes.push_back(tdns::math::Vector3ui(realSizeX, realSizeY, realSizeZ));

            sizeX = (sizeX+1) / 2;
            sizeY = (sizeY+1) / 2;
            sizeZ = (sizeZ+1) / 2;
        }

        //Set the number of levels
        // volumeConfiguration.NbLevels = static_cast<uint32_t>(volumeConfiguration.InitialVolumeSizes.size());
        volumeConfiguration.NbLevels = levelMax;

        // Load the number of bricks of each level
        // for (uint32_t i = 0; i < volumeConfiguration.RealVolumesSizes.size(); ++i)
        // {
        //     tdns::math::Vector3ui nbBricks = volumeConfiguration.RealVolumesSizes[i] / volumeConfiguration.BrickSize;
        //     tdns::math::Vector3ui nbBigBricks = nbBricks / volumeConfiguration.BigBrickSize;

        //     volumeConfiguration.NbBricks.push_back(nbBricks);
        //     volumeConfiguration.NbBigBricks.push_back(nbBigBricks);
        // }

        // // Load the empty brick list if any
        // tinyxml2::XMLNode *emptyBricks = root->FirstChildElement("EmptyBricks");
        // if (emptyBricks)
        // {
        //     element = emptyBricks->FirstChildElement("Brick");
        //     // Load the IDs list of the empty bricks
        //     while (element)
        //     {
        //         // Bkey brickKey = std::stoull(element->Attribute("ID"));
        //         Bkey brickKey = std::stoull(element->Attribute("ID"));
        //         volumeConfiguration.EmptyBricks.push_back(brickKey);

        //         element = element->NextSiblingElement();
        //     }
        // }

        // // Load the volume histogram
        // uint32_t nbBytes = volumeConfiguration.EncodedBytes / volumeConfiguration.Channels;

        // if (nbBytes == 1) // UCHAR1
        //     volumeConfiguration.Histogram.resize(256);
        // else if (nbBytes == 2) // USHORT1
        //     volumeConfiguration.Histogram.resize(65535);
        // tinyxml2::XMLNode *histo = root->FirstChildElement("Histogram");
        // if (histo)
        // {
        //     element = histo->FirstChildElement("Histo");
        //     for (uint32_t i = 0; i < volumeConfiguration.Histogram.size(); ++i)
        //     {
        //         float value = std::stof(element->Attribute("Value"));
        //         volumeConfiguration.Histogram[i] = value;

        //         element = element->NextSiblingElement();
        //     }
        // }

        // LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Bricks.xml file [" << bricksFilePath << "] loaded.");

        return volumeConfiguration;
    }

    void TDNS_API write_volume_configuration(const std::string &file)
    {
    }
} // namespace data
} // namespace tdns