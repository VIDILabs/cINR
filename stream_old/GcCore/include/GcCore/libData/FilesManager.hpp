/*
 */
#pragma once

#include <string>
#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Singleton.hpp>
#include <GcCore/libData/AbstractFile.hpp>

namespace tdns
{
namespace data
{
    class TDNS_API FilesManager : public tdns::common::Singleton<FilesManager>
    {
    public:
        /**
         * Default Constructor.
         */
        FilesManager();

        /**
         * Destructor.
         */
        ~FilesManager();

        /**
         * @brief Get the file from the _rootPath.
         *
         * @param The name of the file in the _rootPath.
         * @return 
         */
        std::unique_ptr<AbstractFile> get_file(const std::string &file);

        /**
         * @brief Get a file from a given path.
         * 
         * @param The complete path of the file to return.
         * @return 
         */
        std::unique_ptr<AbstractFile> get_file_from_path(const std::string &filePath);

    protected:
        /**
        * Member data.
        */
        std::string _rootPath;  ///<
    };
} //namespace data
} //namespace tdns