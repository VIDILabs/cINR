#pragma once

#include <GcCore/libCommon/AbstractTask.hpp>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libNetwork/MessageHandler.hpp>

namespace tdns
{
namespace network
{
    /**
    * @brief Message task, process to message
    */
    class TDNS_API MessageTask : public tdns::common::AbstractTask
    {
    public:
        MessageTask(MessageHandler *handler);

        /**
        * @brief Default destructor.
        */
        virtual ~MessageTask() = default;

        /**
        * @brief do a message task
        */
        virtual void do_task();

    protected:
        MessageHandler      *_handler; ///< Pointer of MessageHandler
    };
} // common
} // tdns