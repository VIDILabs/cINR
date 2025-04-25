#include <GcCore/libNetwork/MessageBoundingBox.hpp>

#include <GcCore/libNetwork/MessageHandler.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace network
{
    const uint32_t MessageBoundingBox::ID = 5;
    
    //---------------------------------------------------------------------------------------------------
    MessageBoundingBox::MessageBoundingBox() : Message()
    {
        uint32_t *id = reinterpret_cast<uint32_t*>(data());
        *id = MessageBoundingBox::ID;

        _bboxmin = reinterpret_cast<tdns::math::Vector3f*>(_data.data() + sizeof(uint32_t));
        _bboxmax = reinterpret_cast<tdns::math::Vector3f*>(_data.data() + sizeof(uint32_t) + sizeof(tdns::math::Vector3f));
    }

    //---------------------------------------------------------------------------------------------------
    MessageBoundingBox::MessageBoundingBox(int8_t *data, size_t size) : Message(data, size)
    { 
        _bboxmin = reinterpret_cast<tdns::math::Vector3f*>(_data.data() + sizeof(uint32_t));
        _bboxmax = reinterpret_cast<tdns::math::Vector3f*>(_data.data() + sizeof(uint32_t) + sizeof(tdns::math::Vector3f));
    }

    //---------------------------------------------------------------------------------------------
    tdns::math::Vector3f& MessageBoundingBox::bounding_box_min_data()
    {
        return *_bboxmin;
    }

    //---------------------------------------------------------------------------------------------
    const tdns::math::Vector3f& MessageBoundingBox::bounding_box_min_data() const
    {
        return *_bboxmin;
    }

    //---------------------------------------------------------------------------------------------
    tdns::math::Vector3f& MessageBoundingBox::bounding_box_max_data()
    {
        return *_bboxmax;
    }

    //---------------------------------------------------------------------------------------------
    const tdns::math::Vector3f& MessageBoundingBox::bounding_box_max_data() const
    {
        return *_bboxmax;
    }
        
    //---------------------------------------------------------------------------------------------
    void MessageBoundingBox::handle(MessageHandler *handler)
    {
        handler->handle(*this);
    }

    //---------------------------------------------------------------------------------------------
    std::unique_ptr<MessageBoundingBox> MessageBoundingBox::create_instance(int8_t *data, size_t size)
    {
        return tdns::common::create_unique_ptr<MessageBoundingBox>(data, size);
    }
} // network
} // tdns