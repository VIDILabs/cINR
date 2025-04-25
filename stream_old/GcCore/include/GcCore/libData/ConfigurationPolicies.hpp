#pragma once

#include <string>
#include <functional>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace data
{
    /**
    * Forward declaration.
    */
    class Configuration;

    /**
    * Structure used to parse the configuration of the 3DNS project.
    */
    struct TDNS_API TDNSConfigurationParser
    {
        static void load_from_file(Configuration &conf, const std::string &file);
    };

    /**
    * Structure used to parse the configuration of the GIS files.
    */
    struct TDNS_API GISConfigurationParser
    {
        static void load_from_file(Configuration &conf, const std::string &file);
    };


} //namespace data
} //namespace tdns