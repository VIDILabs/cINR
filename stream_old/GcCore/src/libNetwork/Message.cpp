#include <GcCore/libNetwork/Message.hpp>

#include <iostream>
#include <bitset>
#include <cstring>

#include <GcCore/libCommon/Memory.hpp>
#include <GcCore/libNetwork/MessageHandler.hpp>

namespace tdns
{
namespace network
{
    const uint32_t Message::ID = 0;

    //---------------------------------------------------------------------------------------------
    Message::Message(int8_t *data, size_t size)
    {
        std::memcpy(_data.data(), data, size);
        
        _size = size;
    } 

    //---------------------------------------------------------------------------------------------
    int8_t* Message::data()
    {
        return _data.data();
    }

    //---------------------------------------------------------------------------------------------
    const int8_t* Message::data() const
    {
        return _data.data();
    }
    
    //---------------------------------------------------------------------------------------------
    std::array<int8_t, 32768>& Message::data_array()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------
    const std::array<int8_t, 32768>& Message::data_array() const
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------
    size_t Message::size() const
    {
        return _size;
    }

    //---------------------------------------------------------------------------------------------
    void Message::handle(MessageHandler *handler)
    {
        handler->handle(*this);
    }

    //---------------------------------------------------------------------------------------------
    std::unique_ptr<Message> Message::create_instance(int8_t *data, size_t size)
    {
        return tdns::common::create_unique_ptr<Message>(data, size);
    }
} // network
} // tdns