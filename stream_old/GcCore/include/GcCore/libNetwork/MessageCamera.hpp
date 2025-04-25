#pragma once

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
    namespace network
    {
        /**
        * @brief MessageCamera for the camera view information from the client to the server
        */
        class TDNS_API MessageCamera : public Message
        {
        public:
            /** Redefine message id */
            static const uint32_t ID;
        public:
            /**
            * @brief default constructor
            */
            MessageCamera();
        
            /**
            * @brief constructor from binary data
            */
            MessageCamera(int8_t *data, size_t size);

            /**
            * @brief Default destructor.
            */
            virtual ~MessageCamera() = default;

            /**
            * 
            */
            float* view_matrix();
            const float* view_matrix() const;

            /**
            * @brief
            *
            * @param[in]    handler
            */
            virtual void handle(MessageHandler *handler);

            /**
            * @brief create an instance of Message with data and size parameters
            *
            * @param[in]    client  client  author
            * @param[in]    data    message data
            * @param[in]    size    message size
            */
            static std::unique_ptr<MessageCamera> create_instance(int8_t *data, size_t size);

        protected:
            /*
            * Member data.
            */
            float    *_viewMatrix; ///< 3x4 view matrix
        };
    } // network
} // tdns