#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <list>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Singleton.hpp>
#include <GcCore/libNetwork/Client.hpp>

namespace tdns 
{
namespace network
{
    /**
    * @brief Client manager class, create client with unique token id
    */
    class TDNS_API ClientManager : public tdns::common::Singleton<ClientManager>
    {
    public:
        /* ClientManager iterator */
        typedef std::map<uint32_t, std::unique_ptr<Client>>::iterator iterator;
    public:
        /**
        * @brief iterator of map
        */
        iterator begin();
        iterator end();

        static ClientManager& get_instance();

        /**
        * @brief create a client from tcpsocket
        *
        * @param[in]    socket      tcp socket
        *
        * @return the new client
        */
        std::unique_ptr<Client> create_client(TcpSocket socket);

        /**
        * @brief add a new client with accept socket
        * /!\ Take the ownership of the client !
        *
        * @param[in]    socket      the accept socket
        */
        void add_client(std::unique_ptr<Client> &client);

        /**
        * @brief get a client from list
        *
        * @param[in]    token_id      the token id
        *
        * @return the new client or null
        */
        Client* get_client(uint32_t token_id);

        /**
        * @brief remove a client from list
        *
        * @param[in] iterator in map
        */
        iterator remove_client(iterator &it);

        /**
        * @brief add new client from list to map (thread safe)
        */
        void handle_new_client();
    protected:
        std::map<uint32_t, std::unique_ptr<Client>> _clients;           ///< All of clients
        std::map<uint32_t, std::unique_ptr<Client>> _newClients;        ///< New client
        std::mutex                                  _accessMutex;       ///< Mutex to map access
    };
} // network
} // tdns