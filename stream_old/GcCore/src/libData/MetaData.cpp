/*
 */
#include <GcCore/libData/MetaData.hpp>

#include <cstdint>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/BricksManager.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libTinyXml/tinyxml2.h>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------
    MetaData::MetaData()
    {
        Configuration &conf = Configuration::get_instance();

        uint32_t nbEncodedBytes;
        conf.get_field("NumberEncodedBytes", nbEncodedBytes);

        uint32_t nbChannels;
        conf.get_field("NumberChannels", nbChannels);

        uint32_t nbBytes = nbEncodedBytes / nbChannels;

        if (nbBytes == 1) // UCHAR1
            _histo.resize(256);
        else if (nbBytes == 2) // USHORT1
            _histo.resize(65535);
    }

    //---------------------------------------------------------------------------------------------
    void MetaData::write_bricks_xml()
    {
        const std::vector<tdns::math::Vector3ui> &levels = _initialLevels;
        Configuration &conf = Configuration::get_instance();
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);
        uint32_t covering;
        conf.get_field("VoxelCovering", covering);
        uint32_t brickSizeWithoutCovering = brickSize - 2 * covering;
        
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLNode *pRoot = doc.NewElement("Meta");
        tinyxml2::XMLNode *pLevels = doc.NewElement("Levels");
        doc.InsertFirstChild(pRoot);
        pRoot->InsertFirstChild(pLevels);

        for (uint32_t i = 0; i < levels.size(); ++i)
        {
            tinyxml2::XMLElement *pElement = doc.NewElement("Level");
            pElement->SetAttribute("level", i);
            pElement->SetAttribute("initial_size_X", levels[i][0]);
            pElement->SetAttribute("initial_size_Y", levels[i][1]);
            pElement->SetAttribute("initial_size_Z", levels[i][2]);
            pElement->SetAttribute("real_size_X", _realLevels[i][0]);
            pElement->SetAttribute("real_size_Y", _realLevels[i][1]);
            pElement->SetAttribute("real_size_Z", _realLevels[i][2]);
            pLevels->InsertEndChild(pElement);
        }

        // write empty bricks if any
        if (_emptyBricks.size() > 0)
        {
            tinyxml2::XMLNode *pEmptyBricks = doc.NewElement("EmptyBricks");
            pRoot->InsertEndChild(pEmptyBricks);

            for (uint32_t i = 0; i < _emptyBricks.size(); ++i)
            {
                tinyxml2::XMLElement *pElement = doc.NewElement("Brick");
                pElement->SetAttribute("ID", static_cast<int64_t>(_emptyBricks[i]));
                pEmptyBricks->InsertEndChild(pElement);
            }
        }

        // write histogram
        tinyxml2::XMLNode *pHisto = doc.NewElement("Histogram");
        pRoot->InsertEndChild(pHisto);
        for (uint32_t i = 0; i < _histo.size(); ++i)
        {
            tinyxml2::XMLElement *pElement = doc.NewElement("Histo");
            pElement->SetAttribute("Value", _histo[i]);
            pHisto->InsertEndChild(pElement);
        }

        std::string volumeDirectory;
        if (!Configuration::get_instance().get_field("VolumeDirectory", volumeDirectory))
        {
            LOGERROR(10, "Unable to get the Volume directory in order to create bricks.xml.");
            return;
        }


        std::string brickFilePath = volumeDirectory
            + BricksManager::get_brick_folder(tdns::math::Vector3ui(brickSize))
            + "/bricks.xml";
        doc.SaveFile(brickFilePath.c_str());
    }

    //---------------------------------------------------------------------------------------------
    bool MetaData::load()
    {
        LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Loading volume metadata.");
        std::string volumeDirectory;

        if (!Configuration::get_instance().get_field("VolumeDirectory", volumeDirectory))
        {
            LOGERROR(10, "Unable to get the Volume directory in order to load the file bricks.xml.");
            return false;
        }

        uint32_t brickSize;
        Configuration::get_instance().get_field("BrickSize", brickSize);

        tdns::math::Vector3ui bigBrickSize;
        Configuration::get_instance().get_field("BigBrickSizeX", bigBrickSize[0]);
        Configuration::get_instance().get_field("BigBrickSizeY", bigBrickSize[1]);
        Configuration::get_instance().get_field("BigBrickSizeZ", bigBrickSize[2]);

        std::string bricksDirectory = volumeDirectory + BricksManager::get_brick_folder(tdns::math::Vector3ui(brickSize));

        std::string bricksFilePath = bricksDirectory + "/bricks.xml";

        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError result = doc.LoadFile(bricksFilePath.c_str());

        if (result != tinyxml2::XML_SUCCESS)
        {
            LOGERROR(10, "Unable to open the file \"" << bricksFilePath << "\". error [" << result << "]");
            return false;
        }

        tinyxml2::XMLNode *root = doc.FirstChild();
        if (!root)
        {
            LOGERROR(10, "No child in the file bricks.xml file. path [" << bricksFilePath << "]");
            return false;
        }

        tinyxml2::XMLNode *levels = root->FirstChildElement("Levels");
        if (!levels)
        {
            LOGERROR(10, "No Levels meta data information in the file bricks.xml file. path [" << bricksFilePath << "]");
            return false;
        }

        // Load intial and real levels sizes
        tinyxml2::XMLElement *element = levels->FirstChildElement("Level");
        while (element)
        {
            uint32_t sizeX = std::atoi(element->Attribute("initial_size_X"));
            uint32_t sizeY = std::atoi(element->Attribute("initial_size_Y"));
            uint32_t sizeZ = std::atoi(element->Attribute("initial_size_Z"));
            _initialLevels.push_back(tdns::math::Vector3ui(sizeX, sizeY, sizeZ));

            sizeX = std::atoi(element->Attribute("real_size_X"));
            sizeY = std::atoi(element->Attribute("real_size_Y"));
            sizeZ = std::atoi(element->Attribute("real_size_Z"));
            _realLevels.push_back(tdns::math::Vector3ui(sizeX, sizeY, sizeZ));
            
            element = element->NextSiblingElement();
        }

        // Load the number of bricks of each level
        LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Metadata : level to load [" << _realLevels.size() << "].");
        for (uint32_t i = 0; i < _realLevels.size(); ++i)
        {
            uint32_t nbBricksX = _realLevels[i][0] / brickSize;
            uint32_t nbBricksY = _realLevels[i][1] / brickSize;
            uint32_t nbBricksZ = _realLevels[i][2] / brickSize;

            uint32_t nbBigBricksX = nbBricksX / bigBrickSize[0];
            uint32_t nbBigBricksY = nbBricksY / bigBrickSize[1];
            uint32_t nbBigBricksZ = nbBricksZ / bigBrickSize[2];

            _nbBricks.push_back(tdns::math::Vector3ui(nbBricksX, nbBricksY, nbBricksZ));
            _nbBigBricks.push_back(tdns::math::Vector3ui(nbBigBricksX, nbBigBricksY, nbBigBricksZ));
        }

        // Load the empty brick list if any
        tinyxml2::XMLNode *emptyBricks = root->FirstChildElement("EmptyBricks"); 
        if (emptyBricks)
        {
            element = emptyBricks->FirstChildElement("Brick");
            // Load the IDs list of the empty bricks
            while (element)
            {
                // Bkey brickKey = std::stoull(element->Attribute("ID"));
                Bkey brickKey = std::stoull(element->Attribute("ID"));
                _emptyBricks.push_back(brickKey);
                
                element = element->NextSiblingElement();
            }
        }

        // Load the volume histogram
        tinyxml2::XMLNode *histo = root->FirstChildElement("Histogram"); 
        if (histo)
        {
            element = histo->FirstChildElement("Histo");
            for (uint32_t i = 0; i < _histo.size(); ++i)
            {
                float value = std::stof(element->Attribute("Value"));
                _histo[i] = value;
                
                element = element->NextSiblingElement();
            }
        }

        LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Bricks.xml file [" << bricksFilePath<< "] loaded.");

        return true;
    }
    
} //namespace data
} //namespace tdns