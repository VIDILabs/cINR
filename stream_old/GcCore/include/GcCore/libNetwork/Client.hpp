#pragma once

#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/Socket.hpp>

namespace tdns
{
namespace network
{
    class Message;

    /**
    * @brief client class, represent one client connection
    */
    class TDNS_API Client
    {
    public:
        /** Constant default size of received data buffer */
        static const uint32_t defaultBufferSize = 2048;
    public:
        /**
        * @brief default constructor, initialized token
        */
        Client(const TcpSocket &socket);

        /**
        * @brief token getter
        *
        * @return the const token
        */
        uint32_t token() const;

        /**
        * @brief tcpSocket getter
        *
        * @return tcp socket
        */
        TcpSocket tcpSocket() const;

        /**
        * @brief send data to a TCP client
        *
        * @param[in]    data    data to send
        * @param[in]    size    size of these data
        *
        * @return error code or length
        */
        int32_t send_data(const int8_t *data, size_t size);

        /**
        * @brief send Message to a TCP client
        *
        * @param[in]    msg    message object to send
        *
        * @return error code or length 
        */
        int32_t send_message(Message *msg);

        /**
        * @brief send data to an UDP client
        *
        * @param[in]    data    data to send
        * @param[in]    size    size of these data
        */
        int32_t broadcast(int8_t *data, size_t size);
        //int32_t broadcast(Message *msg);

        /**
        * @brief receive data from client
        *
        * @param[out]    data    Data received
        */
        int32_t receive(std::vector<int8_t> &data);

    protected:
        /*
        * Member data.
        */
        uint32_t    _token;        ///< token id
        TcpSocket   _tcpSocket;    ///< _tcpsocket object
        //UDPSocket   _udpsocket;
    };
} // network
} // tdns