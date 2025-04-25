#pragma once

#include <map>
#include <functional>
#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Singleton.hpp>
#include <GcCore/libData/AbstractFile.hpp>

namespace tdns
{
namespace data
{
    class TDNS_API FileFactory : public tdns::common::Singleton<FileFactory>
    {
    public:
        /**
         * Default Constructor.
         */
        FileFactory ();

        /**
         * Destructor.
         */
        ~FileFactory();

        /**
         * 
         */
        std::unique_ptr<AbstractFile> create_file(const std::string &extension, const std::string &filePath);

    protected:
        std::map<std::string, std::function<std::unique_ptr<AbstractFile>(const std::string&)>> _map; ///<
    };

} // namespace data
} // namespace tdns