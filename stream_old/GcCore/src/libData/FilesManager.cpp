/*
 */
#include <GcCore/libData/FilesManager.hpp>

#include <string>

#include <GcCore/libData/FileFactory.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libData/Configuration.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    FilesManager::FilesManager()
    {
        // _rootPath = "./data/";
        
        //if (!tdns::common::exists(_rootPath))
        //    create_folder(_rootPath);
    }

    //---------------------------------------------------------------------------------------------------
    FilesManager::~FilesManager()
    {
        ;
    }

    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<AbstractFile> FilesManager::get_file(const std::string &file)
    {
        // Get the file name without the extension
        std::string baseFileName = tdns::common::get_file_base_name(file);
        // Construct the path of the file, from the rootPath
        std::string filePath = baseFileName + "/" + file;
        // std::string filePath = _rootPath + baseFileName + "/" + file;

        return get_file_from_path(filePath);
    }

    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<AbstractFile> FilesManager::get_file_from_path(const std::string &filePath)
    {
        // Get the extension of the file
        std::string extension = tdns::common::get_extension(filePath);

        uint32_t isMultiFiles;
        Configuration::get_instance().get_field("MultipleFiles", isMultiFiles);

        std::string tmpFilePath; // temporary file path to know if it exist or not.
        if (extension == "ima" && isMultiFiles)
        {
            tmpFilePath = tdns::common::remove_extension(filePath) + "_0000.ima";
            extension += "Multi";
        }
        else
            tmpFilePath = filePath;

        // Check if the file exists
        if (!tdns::common::exists(tmpFilePath))
        {
            LOGERROR(10, "Cannot open the file \"" << tmpFilePath << "\". The file does not exists.");
            return nullptr;
        }

        // If the file is a GIS file format (file.ima), it need a header file with the ".dim" extension
        if ((extension == "ima" || extension == "imaMulti") &&
            !tdns::common::exists(tdns::common::remove_extension(filePath) + ".dim"))
        {
            LOGERROR(10, "Cannot open the file \"" << filePath << "\". The corresonding header file .dim does not exists.");
            return nullptr;
        }

        // return the file
        return FileFactory::get_instance().create_file(extension, filePath);
    }
} //namespace data
} //namespace tdns