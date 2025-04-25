#pragma once

#include <string>
#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Check the given file for existence using a read only access
    *
    * @param[in]    path   Path to check.
    *
    * @return True if the file exists and has read access, false otherwise.
    */
    TDNS_API bool exists(const std::string &path);

    /**
    * @brief Indicates if the given path is a directory.
    *
    * @param[in]    path   The path to check for its type.
    *
    * @return True if the path is a directory, false otherwise.
    *              If the path cannot be accessed for privilege reasons, false is returned.
    */
    TDNS_API bool is_dir(const std::string &path);

    /**
    * @brief Indicates if the given path is a regular file.
    *
    * @param[in]    path   The path to check for its type.
    *
    * @return True if the path is a regular file, false otherwise.
    *              If the path cannot be accessed for privilege reasons, false is returned.
    */
    TDNS_API bool is_file(const std::string &path);

    /**
    * @brief Indicates if the given path is a link.
    *
    * @param[in]    path   The path to check for its type.
    *
    * @return True if the path is a link, false otherwise.
    *              If the path cannot be accessed for privilege reasons, false is returned.
    */
    TDNS_API bool is_link(const std::string &path);

    /**
    * @brief Create a folder at the given path, if the 
    *        folder-tree does not exist it will not create it.
    *
    * @param[in]    path   The path to create the folder.
    *
    * @return True if the folder has been created, false otherwise.
    */
    TDNS_API bool create_folder(const std::string &path);

    /**
    * @brief Create a file.
    *
    * @param[in]    path   The path of the file to create.
    */
    TDNS_API void create_file(const std::string &path);

    /**
    * @brief Return the name of a file in a path.
    *
    * @param[in]    path   The path of the file
    *
    * @return A string with the name of a file.
    *
    *       e.g /home/userName/data/myRawFile.raw
    *           will return "myRawFile.raw".
    */
    TDNS_API std::string get_file_name(const std::string &path);

    /**
    * @brief Return the name of a file without his extension.
    *
    * @param[in]    path   The path of the file.
    *
    * @return A string with the base name of a file.
    *
    *       e.g /home/userName/data/myRawFile.raw
    *           will return "myRawFile".
    */
    TDNS_API std::string get_file_base_name(const std::string &path);

    /**
    * @brief Return the extension of a file.
    *
    * @param[in]    path   The path of the file.
    *
    * @return A string with the extension of the file.
    *
    *       e.g /home/userName/data/myRawFile.raw
    *           will return "raw".
    */
    TDNS_API std::string get_extension(const std::string &path);

    /**
    * @brief Return the parent of a file.
    *
    * @param[in]    path    The path of the file.
    *
    * @return A string with the parent of the file.
    *
    *       e.g /home/userName/data/myRawFile.raw
    *           will return "/home/userName/data/"
    */
    TDNS_API std::string get_parent(const std::string &path);

    /**
    * @brief Return the path of a file without the extension.
    *
    * @param[in]    path   The path of the file.
    *
    * @return A string with the file without his extension.
    *
    *       e.g /home/userName/data/myRawFile.raw
    *           will return "/home/userName/data/myRawFile".
    */
    TDNS_API std::string remove_extension(const std::string &path);

    /**
    * @brief Return the number of bytes given in which the encoded
    *        type a volumic data file is encoded.
    *
    * @param[in]    value  The type name of an encoded file.
    *
    * @return The number of bytes.
    *
    *       e.g Possible values are:
    *           - "U8"
    *           - "S8"
    *           - "U16"
    *           - "S16"
    *           - "U32"
    *           - "S32"
    *           - "FLOAT"
    *           - "DOUBLE"
    *           - "RGB"
    *           - "RGBA"
    */
    TDNS_API uint32_t bytes_encoded(const std::string &value);
} // namespace common
} // namespace tdns