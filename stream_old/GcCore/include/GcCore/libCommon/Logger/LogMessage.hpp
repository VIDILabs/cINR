#pragma once

#include <chrono>
#include <sstream>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Logger/LogCommon.hpp>
#include <GcCore/libCommon/Logger/LogCategory.hpp>

namespace tdns
{
namespace common
{
    class TDNS_API LogMessage
    {
    public:
        LogMessage();

        LogMessage(const LogCategory * const cat, log_details::LogType type, log_details::Verbosity verbosity);

        ~LogMessage();

        const std::stringstream& get_streamer() const;
        std::stringstream& get_streamer();

        const std::chrono::system_clock::time_point& get_creation_time() const;
        std::chrono::system_clock::time_point& get_creation_time();

        const std::string& get_subsystem() const;
        log_details::LogType get_type() const;
        log_details::Verbosity get_verbosity() const;

    protected:

        std::stringstream _streamer;
        std::chrono::system_clock::time_point _creationTime;
        const std::string _subSystem;
        const log_details::LogType _type;
        const log_details::Verbosity _verbosity;
    };

    //---------------------------------------------------------------------------------------------
    inline const std::stringstream& tdns::common::LogMessage::get_streamer() const
    {
        return _streamer;
    }

    //---------------------------------------------------------------------------------------------
    inline std::stringstream& tdns::common::LogMessage::get_streamer()
    {
        return _streamer;
    }

    //---------------------------------------------------------------------------------------------
    inline const std::chrono::system_clock::time_point& tdns::common::LogMessage::get_creation_time() const
    {
        return _creationTime;
    }

    //---------------------------------------------------------------------------------------------
    inline std::chrono::system_clock::time_point& tdns::common::LogMessage::get_creation_time()
    {
        return _creationTime;
    }
    //---------------------------------------------------------------------------------------------
    inline const std::string& LogMessage::get_subsystem() const
    {
        return _subSystem;
    }

    //---------------------------------------------------------------------------------------------
    inline log_details::LogType LogMessage::get_type() const
    {
        return _type;
    }

    //---------------------------------------------------------------------------------------------
    inline log_details::Verbosity LogMessage::get_verbosity() const
    {
        return _verbosity;
    }
} // namespace common
} // namespace tdns