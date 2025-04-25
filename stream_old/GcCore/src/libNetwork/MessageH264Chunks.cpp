#include <GcCore/libNetwork/MessageH264Chunks.hpp>

#include <iostream>

#include <GcCore/libNetwork/MessageHandler.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace network
{
    const uint32_t MessageH264Chunks::ID = 2;

    //---------------------------------------------------------------------------------------------------
    MessageH264Chunks::MessageH264Chunks(size_t imageChunkSize) : Message()
    {
        _size = sizeof(uint32_t) + sizeof(uint64_t) + sizeof(size_t) + sizeof(size_t) + imageChunkSize;
        uint32_t *id = reinterpret_cast<uint32_t*>(data());
        *id = MessageH264Chunks::ID;
        init_field();
    }

    //---------------------------------------------------------------------------------------------------
    MessageH264Chunks::MessageH264Chunks(int8_t *data, size_t size) : Message(data, size)
    {
        init_field();
    }

    //---------------------------------------------------------------------------------------------
    uint64_t& MessageH264Chunks::image_id()
    {
        return *_imageId;
    }

    //---------------------------------------------------------------------------------------------
    const uint64_t& MessageH264Chunks::image_id() const
    {
        return *_imageId;
    }

    //---------------------------------------------------------------------------------------------
    size_t& MessageH264Chunks::buffer_offset()
    {
        return *_bufferOffset;
    }

    //---------------------------------------------------------------------------------------------
    const size_t& MessageH264Chunks::buffer_offset() const
    {
        return *_bufferOffset;
    }

    //---------------------------------------------------------------------------------------------
    size_t& MessageH264Chunks::encoded_size()
    {
        return *_encodedSize;
    }

    //---------------------------------------------------------------------------------------------
    const size_t& MessageH264Chunks::encoded_size() const
    {
        return *_encodedSize;
    }

    //---------------------------------------------------------------------------------------------
    uint8_t* MessageH264Chunks::chunk_data()
    {
        return _chunkData;
    }

    //---------------------------------------------------------------------------------------------
    const uint8_t* MessageH264Chunks::chunk_data() const
    {
        return _chunkData;
    }

    //---------------------------------------------------------------------------------------------
    void MessageH264Chunks::handle(MessageHandler *handler)
    {
        handler->handle(*this);
    }

    //---------------------------------------------------------------------------------------------
    std::unique_ptr<MessageH264Chunks> MessageH264Chunks::create_instance(int8_t *data, size_t size)
    {
        return tdns::common::create_unique_ptr<MessageH264Chunks>(data, size);
    }

    //---------------------------------------------------------------------------------------------
    void MessageH264Chunks::init_field()
    {
        int8_t *dataPtr = _data.data() + sizeof(uint32_t);

        _imageId = reinterpret_cast<uint64_t*>(dataPtr);
        dataPtr += sizeof(uint64_t);

        _bufferOffset = reinterpret_cast<size_t*>(dataPtr);
        dataPtr += sizeof(size_t);

        _encodedSize = reinterpret_cast<size_t*>(dataPtr);
        dataPtr += sizeof(size_t);

        _chunkData = reinterpret_cast<uint8_t*>(dataPtr);
    }
} // network
} // tdns