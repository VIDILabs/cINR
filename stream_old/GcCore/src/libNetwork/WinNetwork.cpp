#include <GcCore/libNetwork/WinNetwork.hpp>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/SocketNorm.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------------
#if TDNS_OS == TDNS_OS_WINDOWS
    bool inet_pton(int32_t af, int8_t *src, void *dst)
    {
        struct sockaddr_storage ss;
        int size = sizeof(ss);

        char *cstr = reinterpret_cast<char*>(src);

        // Conversion char* -> LPWSTR
        wchar_t wsrc_copy[INET6_ADDRSTRLEN + 1];
        size_t outSize;
        size_t sizeChar = strlen(cstr) + 1;
        mbstowcs_s(&outSize, wsrc_copy, sizeChar, cstr, sizeChar - 1);
        wsrc_copy[INET6_ADDRSTRLEN] = 0;
        LPWSTR ptr = wsrc_copy;

        if (WSAStringToAddress(ptr, af, NULL, (struct sockaddr *)&ss, &size) == 0) {
            switch (af) {
            case AF_INET:
                *(struct in_addr *)dst = ((struct sockaddr_in *)&ss)->sin_addr;
                return true;
            case AF_INET6:
                *(struct in6_addr *)dst = ((struct sockaddr_in6 *)&ss)->sin6_addr;
                return true;
            }
        }
        return false;
    }
#endif
} // network
} // tdns