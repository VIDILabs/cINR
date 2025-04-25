#include <GcCore/libNetwork/MessageFactory.hpp>

#include <iostream>
#include <bitset>

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libNetwork/MessageConfiguration.hpp>
#include <GcCore/libNetwork/MessageH264Chunks.hpp>
#include <GcCore/libNetwork/MessageCamera.hpp>
#include <GcCore/libNetwork/MessageTransferFunction.hpp>
#include <GcCore/libNetwork/MessageBoundingBox.hpp>
// #include <GcCore/libNetwork/MessageConnection.hpp>
// #include <GcCore/libNetwork/MessagePreprocessing.hpp>
// #include <GcCore/libNetwork/MessageImage.hpp>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------------
    MessageFactory::MessageFactory()
    {
        _map.insert({ Message::ID                   , &Message::create_instance });
        _map.insert({ MessageConfiguration::ID      , &MessageConfiguration::create_instance });
        _map.insert({ MessageH264Chunks::ID         , &MessageH264Chunks::create_instance });
        _map.insert({ MessageCamera::ID             , &MessageCamera::create_instance });
        _map.insert({ MessageTransferFunction::ID   , &MessageTransferFunction::create_instance });
        _map.insert({ MessageBoundingBox::ID        , &MessageBoundingBox::create_instance });
        // _map.insert({ MessageConnection::ID     , &MessageConnection::create_instance });
        // _map.insert({ MessagePreprocessing::ID  , &MessagePreprocessing::create_instance });
        // _map.insert({ MessageImage::ID          , &MessageImage::create_instance });
    }
      
    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<Message> MessageFactory::create_message(int8_t *data, size_t size)
    {
        auto it = _map.find(*(reinterpret_cast<uint32_t*>(data)));
        
        if (it == _map.end())
            return nullptr;
      
        return it->second(data, size);
    }
} // network
} // tdns