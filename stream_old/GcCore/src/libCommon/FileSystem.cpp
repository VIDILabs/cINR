#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

#if TDNS_OS == TDNS_OS_LINUX
#   include <unistd.h>
#   include <sys/types.h>
#   include <sys/stat.h>
#endif

#if TDNS_OS == TDNS_OS_WINDOWS
#   include <Windows.h>
#endif

#include <fstream>
#include <map>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------------
    bool exists(const std::string &path)
    {
#if TDNS_OS == TDNS_OS_LINUX
        return (access(path.c_str(), F_OK) == 0);
#elif TDNS_OS == TDNS_OS_WINDOWS
        DWORD dwAttrib = GetFileAttributes(std::wstring(path.begin(), path.end()).c_str());
        return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
        static_assert(false, "Platform not supported");
#endif
    }

    //---------------------------------------------------------------------------------------------------
    bool is_dir(const std::string &path)
    {
#if TDNS_OS == TDNS_OS_LINUX
        struct stat buf;
        if (stat(path.c_str(), &buf) == 0)
        {
            return S_ISDIR(buf.st_mode);
        }
#elif TDNS_OS == TDNS_OS_WINDOWS
        DWORD dwAttrib = GetFileAttributes(std::wstring(path.begin(), path.end()).c_str());
        return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
        static_assert(false, "Platform not supported");
#endif
        return false;
    }

    //---------------------------------------------------------------------------------------------------
    bool is_file(const std::string &path)
    {
#if TDNS_OS == TDNS_OS_LINUX
        struct stat buf;
        if (stat(path.c_str(), &buf) == 0)
        {
            return S_ISREG(buf.st_mode);
        }
        return false;
#elif TDNS_OS == TDNS_OS_WINDOWS
        DWORD dwAttrib = GetFileAttributes(std::wstring(path.begin(), path.end()).c_str());
        return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
        static_assert(false, "Platform not supported");
#endif
    }

    //---------------------------------------------------------------------------------------------------
    bool is_link(const std::string &path)
    {
#if TDNS_OS == TDNS_OS_LINUX
        struct stat buf;
        if (stat(path.c_str(), &buf) == 0)
        {
            return S_ISLNK(buf.st_mode);
        }
        return false;
#elif TDNS_OS == TDNS_OS_WINDOWS
        DWORD dwAttrib = GetFileAttributes(std::wstring(path.begin(), path.end()).c_str());
        return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_REPARSE_POINT));
#else
        static_assert(false, "Platform not supported");
#endif
    }

    //---------------------------------------------------------------------------------------------------
    bool create_folder(const std::string &path)
    {
#if TDNS_OS == TDNS_OS_LINUX
        return !mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#elif TDNS_OS == TDNS_OS_WINDOWS
        return (CreateDirectory(std::wstring(path.begin(), path.end()).c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError());
#else
        static_assert(false, "Platform not supported");
#endif
    }

    //---------------------------------------------------------------------------------------------------
    void create_file(const std::string &path)
    {
        std::ofstream file(path);
        file.close();
    }

    //---------------------------------------------------------------------------------------------------
    std::string get_file_name(const std::string &path)
    {
        size_t found = path.find_last_of("/\\");
        
        return path.substr(found + 1);
    }

    //---------------------------------------------------------------------------------------------------
    std::string get_file_base_name(const std::string &path)
    {
        size_t foundSlash = path.find_last_of("/\\");
        size_t foundDot = path.find_last_of(".");

        return path.substr(foundSlash + 1, foundDot - (foundSlash + 1));
    }

    //---------------------------------------------------------------------------------------------------
    std::string get_extension(const std::string &path)
    {
        size_t foundDot = path.find_last_of(".");

        return path.substr(foundDot + 1);
    }

    //---------------------------------------------------------------------------------------------------
    std::string get_parent(const std::string &path)
    {
        size_t found = path.find_last_of("/\\");

        return path.substr(0, found + 1);
    }

    //---------------------------------------------------------------------------------------------------
    std::string remove_extension(const std::string &path)
    {
        size_t foundDot = path.find_last_of(".");

        return path.substr(0, foundDot);
    }

    //---------------------------------------------------------------------------------------------------
    uint32_t bytes_encoded(const std::string &value)
    {
        static std::map<std::string, uint32_t> map {
            {"U8",       1},
            {"S8",       1},
            {"U16",      2},
            {"S16",      2},
            {"U32",      4},
            {"S32",      4},
            {"FLOAT",    4},
            {"DOUBLE",   8},
            {"RGB",      3},
            {"RGBA",     4}};

        auto it = map.find(value);

        if (it == map.end())
            return 1;

        return it->second;
    }
} // namespace common
} // namespace tdns
