#pragma once

#include <cstdint>
#include <memory>
#include <array>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace network
{
    class MessageHandler;

    /**
    * @brief class Message
    */
    class TDNS_API Message
    {
    public:
        /** Default message id */
        static const uint32_t ID;

        /** Enumeration of variables types used for fields data */
        enum TYPE { 
            INT8 , UINT8,
            INT16, UINT16,
            INT32, UINT32, FLOAT,
            INT64, UINT64, DOUBLE,
            STRING
        };
    public:

        /**
        * @brief Default constructor.
        */
       Message() = default;

        /**
        * @brief constructor from binary data
        */
        Message(int8_t *data, size_t size);

        /**
        * @brief Default destructor.
        */
        virtual ~Message() = default;
        
        /**
        * @brief getter of data value
        *
        * @return data message
        */
        int8_t* data();
        const int8_t* data() const;

        /**
        * @brief getter of data array
        *
        * @return data message
        */
        std::array<int8_t, 32768>& data_array();
        const std::array<int8_t, 32768>& data_array() const;

        /**
        * @brief getter of data size
        *
        * @return data size
        */
        size_t size() const;
        
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
        static std::unique_ptr<Message> create_instance(int8_t *data, size_t size);
    protected:
        /*
        * Member data.
        */
        std::array<int8_t, 32768>    _data;     ///< array data of message
        size_t                      _size;     ///< size of these data
    };
} // network
} // tdns