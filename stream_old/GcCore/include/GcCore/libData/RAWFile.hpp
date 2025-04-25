/*
 *
 *
 *
 */

#pragma once

#include <string>
#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libData/AbstractFile.hpp>

namespace tdns
{
namespace data
{
    class TDNS_API RAWFile : public AbstractFile
    {
    public:
        /**
        * @brief Constructor.
        *
        * @param path.
        */
        RAWFile(const std::string &filePath, bool readonly = false);

        /**
        * @brief Destructor.
        */
        ~RAWFile();

        /**
         * @brief Open the file.
         */
        virtual void open() override;

        /**
        * @brief Close the file.
        */
        virtual void close() override;

        /**
        * @brief Read size bytes into the file from the
        * current position and put the data in buffer.
        *
        * @param Buffer to store the read data.
        * @param Size the size in bytes to read.
        * @return True if the read worked.
        */
        virtual bool read(uint8_t *buffer, uint32_t size) override;

        /**
        * @brief Write data into the file.
        *
        * @param Data the data to write into the file.
        * @param Size the size in bytes to write into the file.
        * @return True if the write worked.
        */
        virtual bool write(uint8_t *data, uint32_t size) override;

        /**
        * @brief Set the position of the cursor in the file.
        *
        * Set the cursor position from the current position.
        * 
        * @param Offset the offset to add to the current cursor position.
        * @return True if the position set worked.
        */
        virtual bool set_relative_cursor_position(uint64_t offset) override;

        /**
        * @brief Set the absolute position of the file cursor.
        *
        * Set the cursor position from the begining of the file.
        * 
        * @param Position the position of the cursor.
        * @return True if the position set worked.
        */
        virtual bool set_absolute_cursor_position(uint64_t position) override;

        /**
         * @brief Crate an instance of a RAWFile.
         *
         * @param Path of the file.
         * @return
         */
        static std::unique_ptr<RAWFile> create_instance(const std::string &filePath);

    protected:
        /**
        * Member data.
        */
       const bool readonly;
    };
} //namespace data
} //namespace tdns