#include <GcCore/libNetwork/ClientManager.hpp>

#include <iostream>

#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------------
    ClientManager::iterator ClientManager::begin()
    {
        return _clients.begin();
    }

    //---------------------------------------------------------------------------------------------------
    ClientManager::iterator ClientManager::end()
    {
        return _clients.end();
    }

    //---------------------------------------------------------------------------------------------------
    ClientManager& ClientManager::get_instance()
    {
        return tdns::common::Singleton<ClientManager>::get_instance();
    }

    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<Client> ClientManager::create_client(TcpSocket socket)
    {
        return tdns::common::create_unique_ptr<Client>(socket);
    }

    //---------------------------------------------------------------------------------------------------
    void ClientManager::add_client(std::unique_ptr<Client> &client)
    {    
        std::lock_guard<std::mutex> lock(_accessMutex);
        uint32_t token = client.get()->token();
        _newClients[token].swap(client);
    }

    //---------------------------------------------------------------------------------------------------
    Client* ClientManager::get_client(uint32_t token_id)
    {
        auto it = _clients.find(token_id);

        if (it == _clients.end())
            return nullptr;

        return it->second.get();
    }

    //---------------------------------------------------------------------------------------------------
    ClientManager::iterator ClientManager::remove_client(ClientManager::iterator &it)
    {
        return _clients.erase(it);
    }   

    //---------------------------------------------------------------------------------------------------
    void ClientManager::handle_new_client()
    {
        std::lock_guard<std::mutex> lock(_accessMutex);

        uint32_t token;
        for (auto &client : _newClients)
        {
            token = client.first;
            _clients[token].swap(client.second);
        }
        _newClients.clear();
    }
} // network
} // tdns

