#include <GcCore/libNetwork/Client.hpp>

#include <iostream>
#include <string>
#include <bitset>
#include <vector>

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libNetwork/WebsocketHandshake.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------------
    Client::Client(const TcpSocket &socket)
    {
        _tcpSocket = socket;
        _token = static_cast<uint32_t>(socket.socket());
    }   

    //---------------------------------------------------------------------------------------------------
    uint32_t Client::token() const
    {
        return _token;
    }

    //---------------------------------------------------------------------------------------------------
    TcpSocket Client::tcpSocket() const
    {
        return _tcpSocket;
    }

    //---------------------------------------------------------------------------------------------------
    int32_t Client::send_data(const int8_t *data, size_t size)
    {              
        // return _tcpSocket.send(data, size);
        return 0;
    }

    //---------------------------------------------------------------------------------------------------
    int32_t Client::send_message(Message *msg)
    {
        // Encoding data for websocket protocol
        size_t size = msg->size();
        std::vector<int8_t> data;
        WebSocketHandshake::encode_message(msg->data(), size, data);

        return send_data(data.data(), size);
    }

    //---------------------------------------------------------------------------------------------------
    int32_t Client::broadcast(int8_t *data, size_t size)
    {
        return 0;
    }

    //---------------------------------------------------------------------------------------------------
    int32_t Client::receive(std::vector<int8_t> &data)
    {
        // return _tcpSocket.recv(data);
        return 0;
    }
} // network
} // tdns