#include <GcCore/libNetwork/TcpTrait.hpp>

#include <string>
#include <cstring>
#include <iostream>

#include <GcCore/libNetwork/WinNetwork.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace network
{

#if TDNS_OS == TDNS_OS_WINDOWS
    /**
    * @brief redefine inet_pton function to windows
    *
    * @param[in]    af  adress of server
    * @param[in]    src    port of server
    * @param[in]    dst    port of server
    *
    * @return       True if ok
    */
    bool inet_pton(int32_t af, int8_t *src, void *dst);
#endif
    

    //---------------------------------------------------------------------------------------------------
    SOCKET TcpTrait::create_socket_handle()
    {
        return socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    }

    //---------------------------------------------------------------------------------------------------
    struct sockaddr_in TcpTrait::create_socket_address(const std::string &address, int32_t port)
    {
        struct sockaddr_in sockAddress;

        std::memset(&sockAddress, 0, sizeof(struct sockaddr_in));
        sockAddress.sin_family = AF_INET;
        inet_pton(AF_INET, address.c_str(), &(sockAddress.sin_addr));
        sockAddress.sin_port = htons(port);

        return sockAddress;
    }

    //---------------------------------------------------------------------------------------------------
    struct sockaddr_in TcpTrait::create_socket_address(int32_t port)
    {
        struct sockaddr_in sockAddress;

        std::memset(&sockAddress, 0, sizeof(struct sockaddr_in));
        sockAddress.sin_family = AF_INET;
        sockAddress.sin_addr.s_addr = INADDR_ANY;
        sockAddress.sin_port = htons(port);

        return sockAddress;
    }

    //---------------------------------------------------------------------------------------------------
    int32_t TcpTrait::connect_to(SOCKET socket, const std::string &address, int32_t port)
    {
       struct sockaddr_in sockAddress = TcpTrait::create_socket_address(address, port);
       std::cout << "Connect to server : " << address << ":" << port << " ..." << std::endl;
       return connect(socket, reinterpret_cast<struct sockaddr*>(&sockAddress), sizeof(sockAddress));
    }

    //---------------------------------------------------------------------------------------------------
    int32_t TcpTrait::bind_port(SOCKET socket, int32_t port)
    {
       struct sockaddr_in sockAddress = TcpTrait::create_socket_address(port);
       return bind(socket, reinterpret_cast<struct sockaddr*>(&sockAddress), sizeof(sockAddress));
    }

    //---------------------------------------------------------------------------------------------------
    int32_t TcpTrait::send_to(SOCKET socket, const int8_t *data, size_t size)
    {
#if TDNS_OS == TDNS_OS_WINDOWS
        return send(socket, reinterpret_cast<const char*>(data), static_cast<int>(size), NULL);
#else
        return send(socket, reinterpret_cast<const char*>(data), static_cast<int>(size), 0);
#endif
    }

    //---------------------------------------------------------------------------------------------------
    int32_t TcpTrait::recv_from(SOCKET socket, int8_t *data, size_t bufferSize, size_t &sizeRead)
    {
        sizeRead = 0;
        int32_t res;
#if TDNS_OS == TDNS_OS_WINDOWS
        res = recv(socket, reinterpret_cast<char*>(data), static_cast<int>(bufferSize), NULL);
#else
        res = recv(socket, reinterpret_cast<char*>(data), static_cast<int>(bufferSize), 0);
#endif
        sizeRead = static_cast<size_t>(res);
        return res;
    }   
} // network
} // tdns