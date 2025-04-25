#include <GcCore/libNetwork/UdpTrait.hpp>

#include <string>
#include <cstring>

#include <GcCore/libNetwork/WinNetwork.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------------
    SOCKET UdpTrait::create_socket_handle()
    {
        return socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    }

    //---------------------------------------------------------------------------------------------------
    struct sockaddr_in UdpTrait::create_socket_address(int32_t port)
    {
        struct sockaddr_in sockAddress;

        std::memset(&sockAddress, 0, sizeof(struct sockaddr_in));
        sockAddress.sin_family = AF_INET;
        sockAddress.sin_port = htons(port);
        sockAddress.sin_addr.s_addr = htonl(INADDR_ANY);

        return sockAddress;
    }

    //---------------------------------------------------------------------------------------------------
    struct sockaddr_in UdpTrait::create_socket_address(const std::string &address, int32_t port)
    {
        struct sockaddr_in sockAddress;

        std::memset(&sockAddress, 0, sizeof(struct sockaddr_in));
        sockAddress.sin_family = AF_INET;
        sockAddress.sin_port = htons(port);
        inet_pton(AF_INET, address.c_str(), &(sockAddress.sin_addr));
        

        return sockAddress;
    }

    //---------------------------------------------------------------------------------------------------
    int32_t UdpTrait::connect_to(SOCKET socket, const std::string &address, int32_t port)
    {
       struct sockaddr_in sockAddress = UdpTrait::create_socket_address(address, port);
       return connect(socket, reinterpret_cast<struct sockaddr*>(&sockAddress), sizeof(sockAddress));
    }

    //---------------------------------------------------------------------------------------------------
    int32_t UdpTrait::bind_port(SOCKET socket, int32_t port)
    {
       struct sockaddr_in sockAddress = UdpTrait::create_socket_address(port);
       return bind(socket, reinterpret_cast<struct sockaddr*>(&sockAddress), sizeof(sockAddress));
    }

    //---------------------------------------------------------------------------------------------------
    int32_t UdpTrait::send_to(SOCKET socket, const int8_t *data, size_t size, const struct sockaddr_in &sockAddr)
    {
        return sendto(socket,
            reinterpret_cast<const char*>(data),
            static_cast<int>(size),
            0,
            reinterpret_cast<const struct sockaddr*>(&sockAddr),
            static_cast<int>(sizeof(sockAddr)));
    }

    //---------------------------------------------------------------------------------------------------
    int32_t UdpTrait::recv_from(SOCKET socket, int8_t *data, size_t bufferSize, size_t &sizeRead, struct sockaddr_in &sockAddr)
    {
        sizeRead = 0;
        int32_t res;
#if TDNS_OS == TDNS_OS_WINDOWS
        int32_t slen = sizeof(sockAddr);
#else
        uint32_t slen = sizeof(sockAddr);
#endif

        res = recvfrom(socket,
            reinterpret_cast<char*>(data),
            static_cast<int>(bufferSize),
            0,
            reinterpret_cast<struct sockaddr*>(&sockAddr),
            &slen);

        sizeRead = static_cast<size_t>(res);
        return res;
    }
} // network
} // tdns