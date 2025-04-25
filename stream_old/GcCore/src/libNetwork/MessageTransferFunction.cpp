#include <GcCore/libNetwork/MessageTransferFunction.hpp>

#include <GcCore/libNetwork/MessageHandler.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace network
{
    const uint32_t MessageTransferFunction::ID = 4;
    
    //---------------------------------------------------------------------------------------------------
    MessageTransferFunction::MessageTransferFunction() : Message()
    {
        uint32_t *id = reinterpret_cast<uint32_t*>(data());
        *id = MessageTransferFunction::ID;

        _tfData = reinterpret_cast<tdns::math::Vector4f*>(_data.data() + sizeof(uint32_t));
        
        _size = sizeof(uint32_t) + 256 * 4 * sizeof(tdns::math::Vector4f); // ID + 256 * 4 * float
    }

    //---------------------------------------------------------------------------------------------------
    MessageTransferFunction::MessageTransferFunction(int8_t *data, size_t size) : Message(data, size)
    { 
        _tfData = reinterpret_cast<tdns::math::Vector4f*>(_data.data() + sizeof(uint32_t));
    }

    //---------------------------------------------------------------------------------------------
    tdns::math::Vector4f* MessageTransferFunction::transfer_function_data()
    {
        return _tfData;
    }

    //---------------------------------------------------------------------------------------------
    const tdns::math::Vector4f* MessageTransferFunction::transfer_function_data() const
    {
        return _tfData;
    }
        
    //---------------------------------------------------------------------------------------------
    void MessageTransferFunction::handle(MessageHandler *handler)
    {
        handler->handle(*this);
    }

    //---------------------------------------------------------------------------------------------
    std::unique_ptr<MessageTransferFunction> MessageTransferFunction::create_instance(int8_t *data, size_t size)
    {
        return tdns::common::create_unique_ptr<MessageTransferFunction>(data, size);
    }
} // network
} // tdns