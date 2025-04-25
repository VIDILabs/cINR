#pragma once

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
    namespace network
    {
        /**
        * @brief MessageBoundingBox for the bouding box values from the client to the server
        */
        class TDNS_API MessageBoundingBox : public Message
        {
        public:
            /** Redefine message id */
            static const uint32_t ID;
        public:
            /**
            * @brief default constructor
            */
            MessageBoundingBox();
        
            /**
            * @brief constructor from binary data
            */
            MessageBoundingBox(int8_t *data, size_t size);

            /**
            * @brief Default destructor.
            */
            virtual ~MessageBoundingBox() = default;

            /**
            * 
            */
            tdns::math::Vector3f& bounding_box_min_data();
            const tdns::math::Vector3f& bounding_box_min_data() const;

             tdns::math::Vector3f& bounding_box_max_data();
            const tdns::math::Vector3f& bounding_box_max_data() const;


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
            static std::unique_ptr<MessageBoundingBox> create_instance(int8_t *data, size_t size);

        protected:
            /*
            * Member data.
            */
            tdns::math::Vector3f *_bboxmin; ///<
            tdns::math::Vector3f *_bboxmax; ///<
        };
    } // network
} // tdns