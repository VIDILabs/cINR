/*
 *
 */

#pragma once

#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace data
{
    class TDNS_API AbstractFile
    {
    public:
        /**
        * @brief Constructor.
        *
        * @param Path.
        */
        AbstractFile(const std::string &filePath);

        /**
        * @brief Destructor.
        */
        virtual ~AbstractFile();

        /**
        * @brief Open the file.
        */
        virtual void open() = 0;

        /**
        * @brief Close the file.
        */
        virtual void close() = 0;

        /**
        * @brief Read size bytes into the file from the
        * current position and put the data in buffer.
        *
        * @param Buffer to store the read data.
        * @param Size the size in bytes to read.
        * @return True if the read worked.
        */
        virtual bool read(uint8_t *buffer, uint32_t size) = 0;

        /**
        * @brief Write data into the file.
        *
        * @param Data the data to write into the file.
        * @param Size the size in bytes to write into the file.
        * @return True if the write worked.
        */
        virtual bool write(uint8_t *data, uint32_t size) = 0;

        /**
        * @brief Set the position of the cursor in the file.
        *
        * Set the cursor position from the current position.
        * 
        * @param Offset the offset to add to the current cursor position.
        * @return True if the position set worked.
        */
        virtual bool set_relative_cursor_position(uint64_t offset) = 0;

        /**
        * @brief Set the absolute position of the file cursor.
        *
        * Set the cursor position from the begining of the file.
        * 
        * @param Position the position of the cursor.
        * @return True if the position set worked.
        */
        virtual bool set_absolute_cursor_position(uint64_t position) = 0;

    protected:
        /**
        * Member data.
        */
        std::string     _filePath;      ///< Path of the file.
        std::fstream    _fileStream;    ///< Stream of the file.
    };
} //namespace data
} //namespace tdns