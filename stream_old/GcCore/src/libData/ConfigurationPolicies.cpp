#include <GcCore/libData/ConfigurationPolicies.hpp>

#include <string>
#include <fstream>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libTinyXml/tinyxml2.h>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    void TDNSConfigurationParser::load_from_file(Configuration &conf, const std::string &file)
    {
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError result = doc.LoadFile(file.c_str());
        if (result != tinyxml2::XML_SUCCESS)
        {
            LOGERROR(10, "TDNSConfigurationParser: Unable to open the configuration file. error [" << result << "]");
            return;
        }

        tinyxml2::XMLNode *root = doc.FirstChild();
        if (!root)
        {
            LOGERROR(10, " TDNSConfigurationParser: No child in the file");
            return;
        }

        tinyxml2::XMLElement *element = root->FirstChildElement("Field");
        while (element)
        {
            std::string key = element->Attribute("key");
            std::string value = element->Attribute("value");

            conf.add_field(key, value);
            element = element->NextSiblingElement();
        }

        LOGTRACE(10, tdns::common::log_details::Verbosity::INSANE, "TDNSConfigurationParser: configuration loaded.");
    }
    
    //---------------------------------------------------------------------------------------------------
    void GISConfigurationParser::load_from_file(Configuration &conf, const std::string &file)
    {
        std::ifstream infile(file);

        if (!infile)
        {
            LOGERROR(10, "GISConfigurationParser : Unable to open the header file .dim .");
            return;
        }

        std::string size;

        infile >> size;
        conf.add_field("size_X", size);

        infile >> size;
        conf.add_field("size_Y", size);

        infile >> size;
        conf.add_field("size_Z", size);

        // The 4th is not use
        infile >> size;

        std::map<std::string, std::string> map;
        map.insert({"-type", "NumberEncodedBytes"});
        map.insert({"-dx", "dx"});
        map.insert({"-dy", "dy"});
        map.insert({"-dz", "dz"});

        std::string key, value;
        while (infile >> key)
        {   
            infile >> value;
            
            auto it = map.find(key);
            if(it != map.end())
            {
                if (key == "-type")
                    conf.add_field("NumberEncodedBytes", tdns::common::bytes_encoded(value));
                else
                    conf.add_field(it->second, value);
            }
        }

        std::string workingDirectory = "./data/" + tdns::common::get_file_base_name(file) + "/";
        conf.add_field("WorkingDirectory", workingDirectory);
    }
} //namespace data
} //namespace tdns