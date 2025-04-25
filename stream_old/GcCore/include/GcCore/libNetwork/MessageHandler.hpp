#pragma once

#include <string>
#include <sstream>
#include <vector>

#include <GcCore/libCommon/MPMCBoundedQueue.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

#include <GcCore/libNetwork/Message.hpp>
#include <GcCore/libNetwork/MessageConfiguration.hpp>
#include <GcCore/libNetwork/MessageH264Chunks.hpp>
#include <GcCore/libNetwork/MessageCamera.hpp>
#include <GcCore/libNetwork/MessageTransferFunction.hpp>
#include <GcCore/libNetwork/MessageBoundingBox.hpp>
// #include <GcCore/libNetwork/MessagePreprocessing.hpp>
// #include <GcCore/libNetwork/MessageConnection.hpp>
// #include <GcCore/libNetwork/MessageImage.hpp>
// #include <GcCore/libNetwork/MessageLog.hpp>


namespace tdns
{
namespace network
{
    /**
    * @brief
    */
    class TDNS_API MessageHandler
    {
    public:
        /**
        * @brief default constructor
        */
        MessageHandler() = default;

        /**
        * @brief destructor
        */
        virtual ~MessageHandler() = default;
        
        /**
        * @brief
        * 
        * / ! \ It takes the ownership of the object;
        */
        void queue(Message *msg);

        /**
        * @brief treat all message
        */
        void handle_messages();

        /**
        * @brief treat Message object
        * 
        * @param[in]    msg     the message
        */
        virtual void handle(Message &msg);

        /**
        * @brief treat MessageConfiguration object
        *
        * @param[in]    msg     the configuration message
        */
        virtual void handle(MessageConfiguration &msg);

        /**
        * @brief treat MessageH264Chunks object
        *
        * @param[in]    msg     the H264 image chunk message
        */
        virtual void handle(MessageH264Chunks &msg);

        /**
        * @brief treat MessageCamera object
        *
        * @param[in]    msg     the camera message
        */
        virtual void handle(MessageCamera &msg);

        /**
        * @brief treat MessageTransferFunction object
        *
        * @param[in]    msg     the transfer function message
        */
        virtual void handle(MessageTransferFunction &msg);

        /**
        * @brief treat MessageBoundingBox object
        *
        * @param[in]    msg     the bounding box message
        */
        virtual void handle(MessageBoundingBox &msg);

        /**
        * @brief treat MessagePreprocessing object
        *
        * @param[in]    msg     the preprocessing state message
        */
        // virtual void handle(MessagePreprocessing &msg);

        /**
        * @brief treat MessageConnection object
        *
        * @param[in]    msg     the connection message
        */
        // virtual void handle(MessageConnection &msg);

        /**
        * @brief treat MessageCamera object
        *
        * @param[in]    msg     the camera message
        */
        // virtual void handle(MessageImage &msg);

        /**
        * @brief send a log message to client
        *
        * @param[in]    token_id     the token id of client
        * @param[in]    type         the type of log
        * @param[in]    msglog       the message to send
        */
        // virtual void send_log(const uint32_t token_id, MessageLog::TYPE_LOG type, const std::string &msglog);

    private:
        /*
        * Member data.
        */
        tdns::common::MPMCBoundedQueue<Message*> _queue;    ///< Queue containing all message

    };
} // network
} // tdns