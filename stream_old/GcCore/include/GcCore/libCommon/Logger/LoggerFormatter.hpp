#pragma once

#include <chrono>
#include <string>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Logger/LogCommon.hpp>

namespace tdns
{
namespace common
{
    class TDNS_API LoggerFormatter
    {
    public:
        LoggerFormatter(const std::string &headerPatern = "%DATE% @ %TIME_US% %DESC% [%LOGTYPE%] --- ");

        virtual ~LoggerFormatter();

        const std::string get_message_header(const std::string &subsystem, 
                                                const log_details::LogType logType, 
                                                const std::chrono::system_clock::time_point &time);

    protected:

        virtual const std::string compute_header_from_pattern(const std::string &subsystem, 
                                                                const log_details::LogType logType, 
                                                                const std::chrono::system_clock::time_point &time) = 0;
        
    protected:

        std::string _headerPattern;
    };

    //---------------------------------------------------------------------------------------------
    inline LoggerFormatter::LoggerFormatter(const std::string &headerPatern)
    {
        _headerPattern = headerPatern;
    }

    //---------------------------------------------------------------------------------------------
    inline LoggerFormatter::~LoggerFormatter()
    {}

    //---------------------------------------------------------------------------------------------
    inline const std::string LoggerFormatter::get_message_header(const std::string &subsystem, 
                                                                    const log_details::LogType logType, 
                                                                    const std::chrono::system_clock::time_point &time)
    {
        return compute_header_from_pattern(subsystem, logType, time);
    }
} // namespace common
} // namespace tdns
