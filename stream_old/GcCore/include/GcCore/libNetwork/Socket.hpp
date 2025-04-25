#pragma once

#include <vector>
#include <iostream>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/SocketNorm.hpp>
#include <GcCore/libNetwork/TcpTrait.hpp>
#include <GcCore/libNetwork/UdpTrait.hpp>

namespace tdns
{
namespace network
{
    /**
    * @brief socket class, represent a socket
    */
    template<typename Trait>
    class Socket
    {
    public:
        enum Status
        {
            Error = -1,
            Disconnected = 0,
            Partial = 2,
            Done = 3
        };

    public:
        /**
        * @brief Default constructor.
        */
        Socket();

        /**
        * @brief constructor.
        */
        Socket(SOCKET socket);

        /**
        * @brief getter to socket
        *
        * @return socket
        */
        SOCKET socket() const;

        /**
        */
        int32_t connect(const std::string &address, int32_t port);
        int32_t bind(int32_t port);

        /**
        * @brief send data to client
        *
        * @param[in]    data    data to send
        * @param[in]    size    size of data
        *
        * @return error code
        */
        int32_t send(const int8_t *data, size_t size) const;

        /**
        * @brief send data to client
        *
        * @param[in]    data        data to send
        * @param[in]    size        size of data
        * @param[in]    sockAddr    address struct to send
        *
        * @return error code
        */
        int32_t send(const int8_t *data, size_t size, const struct sockaddr_in &sockAddr) const;

        /**
        * @brief Receive data from client
        *
        * @param[out]   data        received data
        * @param[in]    size        size of data
        * @param[in]    sizeRead    Real size read
        *
        * @return error code
        */
        int32_t recv(int8_t *data, size_t bufferSize, size_t &sizeRead);

        /**
        * @brief Receive data from client
        *
        * @param[out]   data        received data
        * @param[in]    size        size of data
        * @param[in]    sizeRead    Real size read
        * @param[in]    sockAddr    address struct to receive
        *
        * @return error code
        */
        int32_t recv(int8_t *data, size_t bufferSize, size_t &sizeRead, struct sockaddr_in &sockAddr);

        /**
        * @brief Receive a compete message from the socket.
        * 
        */
        int32_t recv(std::vector<int8_t> &data);

        /**
        * @brief Receive a compete message from the socket.
        * 
        */
        int32_t recv(std::vector<int8_t> &data, struct sockaddr_in &sockAddr);

    private:
        
        int32_t normalize_result(int32_t result);

    private:
        SOCKET  _socket;    ///< Socket
    };

    typedef Socket<TcpTrait> TcpSocket;
    typedef Socket<UdpTrait> UdpSocket;

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    Socket<Trait>::Socket()
    {
        _socket = Trait::create_socket_handle();
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    Socket<Trait>::Socket(SOCKET socket) : _socket(socket)
    {}

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    inline SOCKET Socket<Trait>::socket() const
    {
        return _socket;
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::connect(const std::string &address, int32_t port)
    {
        return Trait::connect_to(_socket, address, port);
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::bind(int32_t port)
    {
        return Trait::bind_port(_socket, port);
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::send(const int8_t *data, size_t size) const
    {
        return Trait::send_to(_socket, data, size);
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::send(const int8_t *data, size_t size, const struct sockaddr_in &sockAddr) const
    {
        return Trait::send_to(_socket, data, size, sockAddr);
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::recv(int8_t *data, size_t bufferSize, size_t &sizeRead)
    {
        int32_t result = Trait::recv_from(_socket, data, bufferSize, sizeRead);
        
        return normalize_result(result);
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::recv(int8_t *data, size_t bufferSize, size_t &sizeRead, struct sockaddr_in &sockAddr)
    {
        int32_t result = Trait::recv_from(_socket, data, bufferSize, sizeRead, sockAddr);

        return normalize_result(result);
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::recv(std::vector<int8_t> &data)
    {
        size_t iteration = 0, msgSize = 0, sizeRead = 0;
        int32_t result;
        do
        {
            ++iteration;
            data.resize(iteration * 32768);
            result = recv(data.data() + ((iteration - 1) * 32768), data.size(), sizeRead);
            msgSize += sizeRead;
        } while (result == Status::Partial);

        data.resize(msgSize);
        return result;
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::recv(std::vector<int8_t> &data, struct sockaddr_in &sockAddr)
    {
        size_t iteration = 0, msgSize = 0, sizeRead = 0;
        int32_t result;
        do
        {
            ++iteration;
            data.resize(iteration * 32768);
            result = recv(data.data() + ((iteration - 1) * 32768), data.size(), sizeRead, sockAddr);
            msgSize += sizeRead;
        } while (result == Status::Partial);

        data.resize(msgSize);
        return result;
    }

    //---------------------------------------------------------------------------------------------
    template<typename Trait>
    int32_t Socket<Trait>::normalize_result(int32_t result)
    {
        switch (result)
        {
        default:
            return Status::Done;
        case 0:
            return Status::Disconnected;
#if TDNS_OS == TDNS_OS_WINDOWS
        case WSAEMSGSIZE:
            return Status::Partial;
        case SOCKET_ERROR:
            return Status::Error;
#else
//         case MSG_TRUNC:
//             return Status::Partial;
        case -1:
            return Status::Error;
#endif
        }
    }
} // network
} // tdns