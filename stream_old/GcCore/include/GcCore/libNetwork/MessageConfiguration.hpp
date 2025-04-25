#pragma once
#include <map>

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace network
{
    /**
    * @brief MessageConnection for client connection
    */
    class TDNS_API MessageConfiguration : public Message
    {
    public:
        /** Redefine message id */
        static const uint32_t ID;
    public:
        /**
        * @brief constructor to binary data
        */
        MessageConfiguration();

        /**
        * @brief constructor from binary data
        */
        MessageConfiguration(int8_t *data, size_t size);

        /**
        * @brief Default destructor.
        */
        virtual ~MessageConfiguration() = default;

        /**
        * @brief update configuration
        */
        void process();

        /**
        * @brief fille the map of keys
        */
        virtual void fill_map();

        /**
        * @brief get the length of key
        *
        * @param    key_type    the type of key
        *
        * @return the length
        */
        virtual int32_t get_length(int32_t key_type);

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
        static std::unique_ptr<MessageConfiguration> create_instance(int8_t *data, size_t size);

    protected :
        std::map<std::string, int32_t>   _key_id;      ///< map to get the id of field key
        std::map<int32_t, std::string>   _id_key;      ///< map to get the id of field key
        std::map<int32_t, int32_t>       _key_type;    ///< map to get the type of field key
    };
} // network
} // tdns