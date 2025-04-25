#include <GcCore/libCommon/Logger/LogMessage.hpp>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------
    LogMessage::LogMessage() :
        _subSystem("invalid"),
        _type(log_details::LogType::ERROR_TEST),
        _verbosity(log_details::Verbosity::NONE)
    {
        _creationTime = std::chrono::system_clock::now();
    }

    //---------------------------------------------------------------------------------------------
    LogMessage::LogMessage(const LogCategory * const cat, log_details::LogType type, log_details::Verbosity verbosity) :
        _subSystem(cat->get_subsystem()),
        _type(type),
        _verbosity(verbosity)
    {
        _creationTime = std::chrono::system_clock::now();
    }

    //---------------------------------------------------------------------------------------------
    LogMessage::~LogMessage()
    {}
} // namespace common
} // namespace gn