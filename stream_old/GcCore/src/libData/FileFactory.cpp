#include <GcCore/libData/FileFactory.hpp>

#include <GcCore/libData/RAWFile.hpp>
#include <GcCore/libData/GISFile.hpp>
#include <GcCore/libData/GISStackFile.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    FileFactory::FileFactory()
    {
        // insert all the supported file format 
        _map.insert({"ima", &GISFile::create_instance});
        _map.insert({"imaMulti", &GISStackFile::create_instance});
        _map.insert({"raw", &RAWFile::create_instance});
        //_map.insert({"czi", &CZIFile::create_instance});
        //_map.insert({"niftii", &NIFTIIFile::create_instance});
    }

    //---------------------------------------------------------------------------------------------------
    FileFactory::~FileFactory()
    {
        _map.clear();
    }

    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<AbstractFile> FileFactory::create_file(const std::string &extension, const std::string &filePath)
    {
        auto it = _map.find(extension);

        if (it == _map.end())
            return nullptr;

        return it->second(filePath);
    }

} //namespace data
} //namespace tdns