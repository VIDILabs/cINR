#pragma once

#include <list>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Logger/LoggerFormatter.hpp>

namespace tdns
{
namespace common
{
    class TDNS_API LoggerFormatterFileStd : public LoggerFormatter
    {
    protected:
        using PatternList = std::list<std::string>;
        using PatternListIterator = PatternList::iterator;
        using PatternListConstIterator = PatternList::const_iterator;

    public:

        LoggerFormatterFileStd();

        LoggerFormatterFileStd(const std::string &header);

        ~LoggerFormatterFileStd();
            
    protected:

        virtual const std::string compute_header_from_pattern(const std::string &subsystem,
                                                                const log_details::LogType logType,
                                                                const std::chrono::system_clock::time_point &time);

        void process_header_string();

    protected:

        PatternList _patternList;

    };
} // namespace common
} // namespace tdns