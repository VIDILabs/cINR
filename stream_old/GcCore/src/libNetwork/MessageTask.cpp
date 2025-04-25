#include <GcCore/libNetwork/MessageTask.hpp>

#include <iostream>
#include <cstdint>
#include <chrono>
#include <thread>

#include <GcCore/libCommon/Logger/Logger.hpp>

#include <GcCore/libNetwork/Client.hpp>
#include <GcCore/libNetwork/ClientManager.hpp>
#include <GcCore/libNetwork/WebsocketHandshake.hpp>
#include <GcCore/libNetwork/MessageFactory.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------
    MessageTask::MessageTask(MessageHandler *handler) : _handler(handler)
    {}

    //---------------------------------------------------------------------------------------------
    void MessageTask::do_task()
    {
        std::cout << "Message task begin" << std::endl;

        uint32_t res;
        size_t size = 0;

        bool listening = true;

        // Client iterator 
        Client *client = nullptr;
        ClientManager &manager = ClientManager::get_instance();
        
        // Iterator of client manager
        ClientManager::iterator it;

        // FD_SET
        fd_set socket_set;

        while (listening)
        {
            FD_ZERO(&socket_set);

            // Update client list
            manager.handle_new_client();
            ClientManager::iterator it = manager.begin();

            int clientCount = 0, maxFD = 0;
            // Ecoute les donn�es de chaque client
            for (; it != manager.end(); ++it)
            {
                // get current client
                client = it->second.get();
                clientCount++;
                
                // Add to FDSET
                FD_SET(client->tcpSocket().socket(), &socket_set);
                // Set the max
                int currentFD = static_cast<int>(client->tcpSocket().socket());
                maxFD = currentFD > maxFD ? currentFD : maxFD;
            }

            if (clientCount == 0)
                std::this_thread::sleep_for((std::chrono::milliseconds(100)));
            else
            {
                struct timeval timeout = { 0, 100 };
                // if (select(maxFD + 1, &socket_set, NULL, NULL, &timeout) == SOCKET_ERROR)
                if (select(maxFD + 1, &socket_set, NULL, NULL, &timeout) == -1)
                {
#if TDNS_OS == TDNS_OS_WINDOWS
                    std::cerr << "select failed : " << WSAGetLastError() << std::endl;
#elif TDNS_OS == TDNS_OS_LINUX
                    std::cerr << "select failed : " << strerror(errno) << std::endl;
#endif
                    break;
                }
            
                for (it = manager.begin(); it != manager.end(); )
                {
                    // get current client
                    client = it->second.get();
                    
                    if (FD_ISSET(client->tcpSocket().socket(), &socket_set))
                    {
                        std::vector<int8_t> recvbuf;
                        res = client->receive(recvbuf);
                        if (res > 0)
                        {
                            std::cout << "received data from client " << client->token() << std::endl;

                            size = recvbuf.size();
                            int8_t *decodedMsg = WebSocketHandshake::decode_message(recvbuf.data(), size);

                            // Add message to handler
                            std::unique_ptr<Message> msg = MessageFactory::get_instance().create_message(decodedMsg, size);
                            ++it;// before the continue

                            if (!msg) continue;
                            _handler->queue(msg.release());
                        }
                        else if (res == 0)
                        {
                            it = manager.remove_client(it);
                            FD_CLR(client->tcpSocket().socket(), &socket_set);
                            std::cerr << "Connection closed" << std::endl;
                        }
                        else
                        {
#if TDNS_OS == TDNS_OS_WINDOWS
                            std::cerr << "WebsocketServer::listen_data : " << WSAGetLastError() << std::endl;
#elif TDNS_OS == TDNS_OS_LINUX
                            std::cerr << "WebsocketServer::listen_data : " << strerror(errno) << std::endl;
#endif
                            listening = false;
                            throw std::runtime_error(""); //< mettre un message
                        }
                    }
                    else                    
                        ++it;                    
                }
            }
        }

        LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Listen data end thread.");
    }
} // common
} // tdns