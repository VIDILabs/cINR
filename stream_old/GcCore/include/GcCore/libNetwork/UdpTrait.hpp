#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/SocketNorm.hpp>

namespace tdns
{
namespace network
{
    struct TDNS_API UdpTrait
    {
        /** Constant default size of received data buffer */
        static const uint32_t defaultBufferSize = 1500;

        /**
        */
        static SOCKET create_socket_handle();

        /**
        */
        static struct sockaddr_in create_socket_address(int32_t port);
        static struct sockaddr_in create_socket_address(const std::string &address, int32_t port);

        /**
        */
        static int32_t connect_to(SOCKET socket, const std::string &address, int32_t port);

        /**
        */
        static int32_t bind_port(SOCKET socket, int32_t port);

        /**
        * @brief send data to a client
        *
        * @param[in]    client      the udp client
        * @param[in]    data        data to send
        * @param[in]    size        size of data
        * @param[in]    sockAddr    address struct to send
        */
        static int32_t send_to(SOCKET socket, const int8_t *data, size_t size, const struct sockaddr_in &sockAddr);

        /**
        * @brief receive data from a client
        *
        * @param[in]    socket      The socket that will be read.
        * @param[out]   data        Data received.
        * @param[in]    bufferSize  Maximum size of the buffer.
        * @param[in]    sizeRead    Real size read
        * @param[in]    sockAddr    address struct to receive
        */
        static int32_t recv_from(SOCKET socket, int8_t *data, size_t bufferSize, size_t &sizeRead, struct sockaddr_in &sockAddr);
    };
} // network
} // tdns