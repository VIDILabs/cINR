#include <sstream>
#include <chrono>
#include <ctime>

#include <GcCore/libCommon/Logger/LoggerFormatterFileStd.hpp>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------
    LoggerFormatterFileStd::LoggerFormatterFileStd() : LoggerFormatter()
    {
        process_header_string();
    }

    //---------------------------------------------------------------------------------------------
    LoggerFormatterFileStd::LoggerFormatterFileStd(const std::string &header) : LoggerFormatter(header)
    {
        process_header_string();
    }

    //---------------------------------------------------------------------------------------------
    LoggerFormatterFileStd::~LoggerFormatterFileStd()
    {}

    //---------------------------------------------------------------------------------------------
    const std::string LoggerFormatterFileStd::compute_header_from_pattern(const std::string &subsystem,
                                                                            const log_details::LogType logType,
                                                                            const std::chrono::system_clock::time_point &time)
    {
        std::stringstream stream;
        std::time_t atime = std::chrono::system_clock::to_time_t(time);
        std::tm atm = *(std::localtime(&atime));
        std::chrono::system_clock::time_point badRes = std::chrono::system_clock::from_time_t(atime);

        for (PatternListIterator it = _patternList.begin(); it != _patternList.end(); ++it)
        {

            if (*it == "%DATE%")
            {
                stream << atm.tm_year + 1900 << "/";
                if (atm.tm_mon + 1 < 10)
                {
                    stream << '0';
                }
                stream << atm.tm_mon + 1 << "/";
                if (atm.tm_mday < 10)
                {
                    stream << '0';
                }
                stream << atm.tm_mday;
            }
            else if (*it == "%TIME_US%")
            {
                stream << atm.tm_hour << ":";
                if (atm.tm_min < 10)
                {
                    stream << '0';
                }
                stream << atm.tm_min << ":";
                if (atm.tm_sec < 10)
                {
                    stream << '0';
                }
                stream << atm.tm_sec << ".";

                stream.width(6);
                stream.fill('0');
                stream << std::chrono::duration_cast<std::chrono::microseconds>(time - badRes).count();
            }
            else if (*it == "%DESC%")
            {
                stream << subsystem;
            }
            else if (*it == "%LOGTYPE%")
            {
                static char msgType[] = {'D', 'T', 'I', 'W', 'E', 'F'};
                stream << msgType[logType];
            }
            else
            {
                stream << *it;
            }
        }

        return stream.str();
    }

    //---------------------------------------------------------------------------------------------
    void LoggerFormatterFileStd::process_header_string()
    {
        _patternList.clear();
        std::string currentToken;

        for (size_t i = 0; i < _headerPattern.size(); )
        {
            switch (_headerPattern[i])
            {
            case '%':
            {
                if(currentToken.size() != 0)
                {
                    _patternList.push_back(currentToken);
                    currentToken.clear();
                }

                size_t end = _headerPattern.find('%', i + 1);
                std::string token = _headerPattern.substr(i, end - i + 1);
                _patternList.push_back(token);
                i += token.size();
                break;
            }
            default:
                currentToken += _headerPattern[i];
                ++i;
            }
        }

        if (currentToken.size() != 0)
        {
            _patternList.push_back(currentToken);
            currentToken.clear();
        }
    }
} // namespace common
} // namespace tdns