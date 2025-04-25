/*
 */

#include <GcCore/libData/AbstractFile.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    AbstractFile::AbstractFile(const std::string &filePath)
    {
        _filePath = filePath;
    }

    //---------------------------------------------------------------------------------------------------
    AbstractFile::~AbstractFile()
    {
        ;
    }
} //namespace data
} //namespace tdns
