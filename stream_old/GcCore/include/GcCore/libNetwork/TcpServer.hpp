#pragma once

#include <cstdint>
#include <set>
#include <cstring>
#include <thread>
#include <mutex>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/SocketNorm.hpp>

namespace tdns
{
namespace network
{
    class Client;
    /**
    * @brief TcpServer class, that usefull to communicate via TCP socket
    */
    class TDNS_API TcpServer
    {
    public:
        /**
        * @brief : constructor with port parameter listen on localhost
        *
        * @param[in]    port    server port
        */
        TcpServer(uint32_t port);

        /**
        * @brief : constructor with port parameter listen on localhost
        *
        * @param[in]    port    server port
        * @param[in]    adresse server adress
        */
        TcpServer(uint32_t port, const std::string &address);

        /**
        * @brief : destructor
        */
        ~TcpServer();       

        /**
        * @brief run tcpserver (listen all connections)
        */
        virtual void run();
        
        /**
        * @brief stop listening and server
        */
        virtual void stop();
    public:
        /**
        * @brief create socket adress
        *
        * @param[in]    adress  adress of server
        * @param[in]    port    port of server
        *
        * @return       socket adress structure
        */
        static inline struct sockaddr_in create_socket_adress(int8_t *adress, int32_t port)
        {
            struct sockaddr_in sockAdress;

            std::memset(&sockAdress, 0, sizeof(struct sockaddr_in));
            sockAdress.sin_family = AF_INET;
#if TDNS_OS == TDNS_OS_WINDOWS
            inet_pton(AF_INET, adress, &(sockAdress.sin_addr));
#else
            inet_pton(AF_INET, reinterpret_cast<char*>(adress), &(sockAdress.sin_addr));
#endif
            sockAdress.sin_port = htons(port);

            return sockAdress;
        }

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
        static inline bool inet_pton(int32_t af, int8_t *src, void *dst)
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
    protected:
        /**
        * @brief initialize tcpserver
        */
        void init();
    protected:
        /*
        * Member data.
        */
        uint32_t            _port;    ///< Port of tcp server
        std::string         _address; ///< Address of tcp server    
    };
} //namespace network
} //namespace tdns
