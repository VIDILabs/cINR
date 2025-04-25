#include <GcCore/libNetwork/MessageHandler.hpp>

#include <iostream>

namespace tdns
{
namespace network
{
    //---------------------------------------------------------------------------------------------------
    void MessageHandler::queue(Message *msg)
    {
        _queue.push(msg);
    }

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle_messages()
    {
        Message *msg;
        while (!_queue.empty())
        {
            if (!_queue.pop(msg)) continue;
            
            msg->handle(this);
            delete msg;
        }
    }

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle(Message &msg)
    {
        // std::cout << "New message" << std::endl;
    }

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle(MessageConfiguration &msg)
    {}

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle(MessageH264Chunks &msg)
    {}

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle(MessageCamera &msg)
    {}

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle(MessageTransferFunction &msg)
    {}

    //---------------------------------------------------------------------------------------------------
    void MessageHandler::handle(MessageBoundingBox &msg)
    {}

    // //---------------------------------------------------------------------------------------------------
    // void MessageHandler::handle(MessagePreprocessing &msg)
    // {}

    // //---------------------------------------------------------------------------------------------------
    // void MessageHandler::handle(MessageConnection &msg)
    // {}

    // //---------------------------------------------------------------------------------------------------
    // void MessageHandler::handle(MessageImage &msg)
    // {}

    // //---------------------------------------------------------------------------------------------------
    // void MessageHandler::send_log(const uint32_t token_id, MessageLog::TYPE_LOG type, const std::string &msglog)
    // {
    //     MessageLog msg(token_id, type, msglog);
    //     ClientManager::get_instance()->get_client(token_id)->send_message(&msg);
    // }
} // network
} // 