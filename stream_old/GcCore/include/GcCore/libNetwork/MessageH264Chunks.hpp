#pragma once

#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/Message.hpp>

namespace tdns
{
namespace network
{
    /**
    * @brief Message for chunks of H264 encoded images
    */
    class TDNS_API MessageH264Chunks : public Message
    {
    public:
        /** Redefine message id */
        static const uint32_t ID;
    public:

        /**
        * @brief default constructor
        */
        MessageH264Chunks(size_t imageChunkSize);

        /**
        * @brief default constructor (call from create instance)
        */
        MessageH264Chunks(int8_t *data, size_t size);

        /**
        * @brief Default destructor.
        */
        virtual ~MessageH264Chunks() = default;

        /**
        */
        uint64_t& image_id();
        const uint64_t& image_id() const;

        /**
        */
        size_t& buffer_offset();
        const size_t& buffer_offset() const;

        /**
        */
        uint64_t& encoded_size();
        const uint64_t& encoded_size() const;

        /**
        */
        uint8_t* chunk_data();
        const uint8_t* chunk_data() const;

        /**
        * @brief
        *
        * @param[in]    handler
        */
        virtual void handle(MessageHandler *handler) override;

        /**
        * @brief create an instance of Message with data and size parameters
        *
        * @param[in]    data    message data
        * @param[in]    size    message size
        */
        static std::unique_ptr<MessageH264Chunks> create_instance(int8_t *data, size_t size);

    private:

        void init_field();

    protected:
        /*
        * Member data.
        */
        uint64_t    *_imageId;
        size_t      *_bufferOffset;
        size_t      *_encodedSize;
        uint8_t     *_chunkData;
    };
} // network
} // tdns