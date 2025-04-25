#pragma once

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
    namespace network
    {
        /**
        * @brief MessageTransferFunction for the transfer function values from the client to the server
        */
        class TDNS_API MessageTransferFunction : public Message
        {
        public:
            /** Redefine message id */
            static const uint32_t ID;
        public:
            /**
            * @brief default constructor
            */
            MessageTransferFunction();
        
            /**
            * @brief constructor from binary data
            */
            MessageTransferFunction(int8_t *data, size_t size);

            /**
            * @brief Default destructor.
            */
            virtual ~MessageTransferFunction() = default;

            /**
            * 
            */
            tdns::math::Vector4f* transfer_function_data();
            const tdns::math::Vector4f* transfer_function_data() const;

            /**
            * @brief
            *
            * @param[in]    handler
            */
            virtual void handle(MessageHandler *handler);

            /**
            * @brief create an instance of Message with data and size parameters
            *
            * @param[in]    data    message data
            * @param[in]    size    message size
            */
            static std::unique_ptr<MessageTransferFunction> create_instance(int8_t *data, size_t size);

        protected:
            /*
            * Member data.
            */
            tdns::math::Vector4f *_tfData; ///< 
        };
    } // network
} // tdns