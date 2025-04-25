#include <GcCore/libData/GISStackFile.hpp>

#include <GcCore/libCommon/Memory.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libData/Configuration.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    GISStackFile::GISStackFile(const std::string &filePath)
        : AbstractFile(filePath), _currentPositionInFile(0), _currentSlice(0)
    {
        std::string headerFilePath = tdns::common::remove_extension(filePath) + ".dim";
        _filePathNoExtension = tdns::common::remove_extension(filePath);

        Configuration &config = Configuration::get_instance();
        config.load<tdns::data::GISConfigurationParser>(headerFilePath);

        uint64_t volumeSizeX, volumeSizeY, volumeSizeZ;
        config.get_field("size_X", volumeSizeX);
        config.get_field("size_Y", volumeSizeY);
        config.get_field("size_Z", volumeSizeZ);

        uint64_t encoded;
        config.get_field("NumberEncodedBytes", encoded);
        _sliceDimensionByte = volumeSizeX * volumeSizeY * encoded;
    }

    //---------------------------------------------------------------------------------------------------
    GISStackFile::~GISStackFile()
    {
        if (_fileStream.is_open())
            close();
    }

    //---------------------------------------------------------------------------------------------------
    void GISStackFile::open()
    {
        constexpr size_t n_zero = 4;
        std::string fileNumber = std::string(n_zero - std::to_string(_currentSlice).size(), '0') + std::to_string(_currentSlice);
        std::string filePath = _filePathNoExtension + "_" + fileNumber + ".ima";
        if (!tdns::common::exists(filePath))
            tdns::common::create_file(filePath);

        _fileStream.open(filePath, std::fstream::in | std::fstream::out | std::fstream::binary);
    }

    //---------------------------------------------------------------------------------------------------
    void GISStackFile::close()
    {
        _fileStream.close();
    }

    //---------------------------------------------------------------------------------------------------
    bool GISStackFile::read(uint8_t *buffer, uint32_t size)
    {
        if (!_fileStream.is_open())
        {
            LOGERROR(10, "Cannot read the file ! File is not opened.");
            return false;
        }

        //case where we are in at the end of the file.
        if (_currentPositionInFile + size > _sliceDimensionByte)
        {
            //read the end of the file.
            size_t size_to_read = 0;
            size_to_read = _sliceDimensionByte - _currentPositionInFile;
            _fileStream.read(reinterpret_cast<char*>(buffer), size_to_read);
            
            //load the new file.
            this->close();
            ++_currentSlice;
            this->open();

            //move and read the rest.
            buffer += size_to_read;
            size_to_read = size - size_to_read;
            _fileStream.read(reinterpret_cast<char*>(buffer), size_to_read);
        }
        else
        {
            _fileStream.read(reinterpret_cast<char*>(buffer), size);
        }

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool GISStackFile::write(uint8_t *data, uint32_t size)
    {
        LOGERROR(10, "GISStackFile::write not implemented.");
        return false;
    }

    //---------------------------------------------------------------------------------------------------
    bool GISStackFile::set_relative_cursor_position(uint64_t offset)
    {
        LOGERROR(10, "GISStackFile::set_relative_cursor_position not implemented.");
        return false;
    }

    //---------------------------------------------------------------------------------------------------
    bool GISStackFile::set_absolute_cursor_position(uint64_t position)
    {
        uint64_t expectedSlice = position / _sliceDimensionByte;
        uint64_t expectedPositionInSlice = position % _sliceDimensionByte;

        //change open 
        if (expectedSlice != _currentSlice)
        {
            this->close();
            _currentSlice = expectedSlice;
            this->open();
        }

        if (!_fileStream.is_open())
        {
            LOGERROR(10, "Cannot set the cursor position in the file ! File is not opened.");
            return false;
        }

        _currentPositionInFile = expectedPositionInSlice;
        _fileStream.seekg(expectedPositionInSlice);

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<GISStackFile> GISStackFile::create_instance(const std::string &filePath)
    {
        return tdns::common::create_unique_ptr<GISStackFile>(filePath);
    }
} //namespace data
} //namespace tdns