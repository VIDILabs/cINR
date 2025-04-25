#include <GcCore/libNetwork/TcpServer.hpp>

#include <iostream>
#include <string>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------
    TcpServer::TcpServer(uint32_t port) : TcpServer(port, "127.0.0.1")
    {}

    //---------------------------------------------------------------------------------------------
    TcpServer::TcpServer(uint32_t port, const std::string &address) :  _port(port), _address(address)
    {
        //LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Initialize tcp server on port [" << port << "] at [" << adress << "]");
        std::cout << "Initialize tcp server on port [" << port << "] at [" << address << "]" << std::endl;
        init();
        std::cout << "Server initialized" << std::endl;
        //LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Server initialized");
    }

    //---------------------------------------------------------------------------------------------
    TcpServer::~TcpServer()
    {}

    //---------------------------------------------------------------------------------------------
    void TcpServer::init()
    {
//#if TDNS_OS == TDNS_OS_WINDOWS
//        WSADATA wsa_data;
//        WSAStartup(MAKEWORD(1, 1), &wsa_data);
//#endif
//        // Create server socket
//        std::cout << "Create socket..." << std::endl;
//        //LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Create socket...");
//        if ((_serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1)
//        {
//            //LOGFATAL(50, "Create socket failed.");
//            std::cout << "Create socket failed." << std::endl;
//            return;
//        }
//
//        // Bind socket
//        //LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Bind socket...");
//        std::cout << "Bind socket..." << std::endl;
//        struct sockaddr_in sockAdress = create_socket_adress(_address, _port);
//        if (bind(_serverSocket, (struct sockaddr*)&sockAdress, sizeof(struct sockaddr_in)) == -1)
//        {
//            //LOGFATAL(50, "Bind socket failed.");
//            std::cout << "Bind socket failed." << std::endl;
//            return;
//        }       
//
//        // Listen connexion
//        if (listen(_serverSocket, 1) == -1)
//        {
//            //LOGFATAL(50, "Listen failed.");
//            std::cout << "Listen failed." << std::endl;
//            _listening = false;
//        }
    }

    //---------------------------------------------------------------------------------------------
    void TcpServer::run()
    { 
        //LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Listen PORT [" << _port << "] ...");
        std::cout << "Listen PORT [" << _port << "] ..." << std::endl;        
    }

    //---------------------------------------------------------------------------------------------
    //void TcpServer::listen_connection()
    //{
        //SOCKET currentSocketClient;

        //_stopServerMutex.lock();
        //bool listening = _listening;
        //_stopServerMutex.unlock();

        //// new client message buffer
        ////int8_t newClientMsg[8];

        //while (listening)
        //{       
        //    // new client
        //    if ((currentSocketClient = accept(_serverSocket, NULL, NULL)) == -1)
        //    {
        //        if (errno != EINTR)
        //        {
        //            //LOGFATAL(50, "Accept tcp client failed.");
        //            std::cout << "Accept tcp client failed." << std::endl;
        //            _listening = false;
        //        }
        //    }

        //    // Create message
        //    /*uint32_t *newClientMsg32 = reinterpret_cast<uint32_t*>(newClientMsg);
        //    *(newClientMsg32) = MessageConnection::ID;
        //    *(newClientMsg32 + 1) = currentSocketClient;
        //    
        //     // Add message to handler
        //    std::unique_ptr<Message> msg = MessageFactory::get_instance()->create_message(newClientMsg, 8 * sizeof(int8_t));
        //    if (!msg) continue;
        //    MessageHandler::get_instance()->queue(msg.release());*/
        //    
        //    std::lock_guard<std::mutex> guard(_stopServerMutex);
        //    listening = _listening;
        //}

        ////LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Close tcp server.");
        //std::cout << "listen connection end thread." << std::endl;
    //}

    //---------------------------------------------------------------------------------------------
    //void TcpServer::listen_data()
    //{
        //uint32_t res;
        //int8_t recvbuf[1025];
        //int8_t receivedMsg[1029];

        //_stopServerMutex.lock();
        //bool listening = _listening;
        //_stopServerMutex.unlock();

        //while (listening)
        //{
        //    std::lock_guard<std::mutex> guardClient(_clientsMutex);
        //    // Ecoute les donn�es de chaque client
        //    for (Client *client : _clients)
        //    {
        //        res = client->receive(recvbuf, 1024);
        //        if (res > 0)
        //        {
        //            recvbuf[res] = '\0';
        //            std::cout << recvbuf << std::endl;

        //            // Add message to handler
        //            sprintf_s(reinterpret_cast<char*>(receivedMsg), res + 4, "%u%s", Message::ID, recvbuf);

        //            // Add message to handler
        //            std::unique_ptr<Message> msg = MessageFactory::get_instance()->create_message(receivedMsg, res);
        //            if (!msg) continue;
        //            //MessageHandler::get_instance()->queue(msg.release());
        //        }
        //        else if (res == 0)
        //            std::cerr << "Connection closed" << std::endl;
        //        else
        //            std::cerr << "websocket::create receive()" << WSAGetLastError() << std::endl;
        //    }

        //    std::lock_guard<std::mutex> guardListening(_stopServerMutex);
        //    bool listening = _listening;
        //}

        ////LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Close tcp server.");
        //std::cout << "listen data end thread." << std::endl;
    //}

    //---------------------------------------------------------------------------------------------
    void TcpServer::stop()
    {

        //LOGTRACE(50, tdns::common::log_details::Verbosity::INSANE, "Close tcp server.");
        std::cout << "Close tcp server." << std::endl;
    }  
} //namespace network
} //namespace tdns
