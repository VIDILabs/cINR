#include <GcCore/libNetwork/MessageCamera.hpp>

#include <GcCore/libNetwork/MessageHandler.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace network
{
    const uint32_t MessageCamera::ID = 3;
    
    //---------------------------------------------------------------------------------------------------
    MessageCamera::MessageCamera() : Message()
    {
        uint32_t *id = reinterpret_cast<uint32_t*>(data());
        *id = MessageCamera::ID;

        _viewMatrix = reinterpret_cast<float*>(_data.data() + sizeof(uint32_t));
        
        _size = sizeof(uint32_t) + 12 * sizeof(float); // ID + 3*4 float
    }

    //---------------------------------------------------------------------------------------------------
    MessageCamera::MessageCamera(int8_t *data, size_t size) : Message(data, size)
    { 
        _viewMatrix = reinterpret_cast<float*>(_data.data() + sizeof(uint32_t));
    }

    //---------------------------------------------------------------------------------------------
    float* MessageCamera::view_matrix()
    {
        return _viewMatrix;
    }
        
    //---------------------------------------------------------------------------------------------
    void MessageCamera::handle(MessageHandler *handler)
    {
        handler->handle(*this);
    }

    //---------------------------------------------------------------------------------------------
    std::unique_ptr<MessageCamera> MessageCamera::create_instance(int8_t *data, size_t size)
    {
        return tdns::common::create_unique_ptr<MessageCamera>(data, size);
    }
} // network
} // tdns