#pragma once

#include <map>
#include <functional>
#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Singleton.hpp>

namespace tdns
{
namespace network
{
    class Message;
    /**
    * @brief class Message
    */
    class TDNS_API MessageFactory : public tdns::common::Singleton<MessageFactory>
    {
    public:
        /**
        * @brief default constructor, initialize all message types
        */
        MessageFactory();

        /**
        */
        std::unique_ptr<Message> create_message(int8_t *data, size_t size);

    protected:
        /*
        * Member data.
        */
        std::map<uint32_t, std::function<std::unique_ptr<Message>(int8_t*, size_t)>>  _map;   ///< Function pointer to treat websocket client connection
    };
} // network
} // tdns