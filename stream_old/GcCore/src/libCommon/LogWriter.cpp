#include <GcCore/libCommon/Logger/LogWriter.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/LogCommon.hpp>
#include <GcCore/libCommon/Logger/LogMessage.hpp>
#include <GcCore/libCommon/Logger/LoggerFormatter.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------
    LogWriter::LogWriter()
    {
        _isMt = false;
        _run = false;
    }

    //---------------------------------------------------------------------------------------------
    LogWriter::~LogWriter()
    {
        if (_isMt)
        {
            stop_thread();
        }

        close_stream();
    }

#define IO_DT_FORMAT(data, width) \
    std::setw(width) << std::setfill('0') << data
    //---------------------------------------------------------------------------------------------
    bool LogWriter::open_stream(const std::string &filePath, LoggerFormatter &formatter, bool isMt)
    {
        if (is_open())
        {
            LOGERROR(0, "LogWriter::open_stream : a stream is already open!");
            return false;
        }

        _formatter = &formatter;

        // check the output file
        // std::chrono::system_clock::time_point today = std::chrono::system_clock::now();
        // std::time_t aTime = std::chrono::system_clock::to_time_t(today);
        //std::tm aCalTime = *(std::localtime(&aTime));

        std::stringstream stream;
        stream << filePath
#if TDNS_OS == TDNS_OS_WINDOWS
#   pragma message("Log file name has been changed ! < ")
#elif TDNS_OS == TDNS_OS_LINUX
#   warning "Log file name has been changed ! <"
#endif
            //<< '_'
            //<< IO_DT_FORMAT(1900 + aCalTime.tm_year, 4)
            //<< IO_DT_FORMAT(1 + aCalTime.tm_mon, 2)
            //<< IO_DT_FORMAT(aCalTime.tm_mday, 2)
            //<< '_'
            //<< IO_DT_FORMAT(aCalTime.tm_hour, 2)
            //<< IO_DT_FORMAT(aCalTime.tm_min, 2)
            //<< IO_DT_FORMAT(aCalTime.tm_sec, 2)
            << ".log";

        const std::string path(stream.str());

//        if (tdns::common::exists(path))
//        {
//#if TDNS_MODE == TDBS_MODE_DEBUG
//            std::cout << "Logger::open_stream : Error : [" << path << "] already exists!" << std::endl;
//#endif
//            return false;
//        }

        _file.open(path.c_str(), std::ios::trunc);

        const bool fileResult = is_open();
        if (!fileResult)
        {
            LOGERROR(0, "LogWritter::open_stream : file not opened!");
            return false;
        }

        // init thread part
        _isMt = isMt;
        if (_isMt)
        {
            _run = true;
            start_thread();
        }

        return fileResult;
    }
#undef IO_DT_FORMAT

    //---------------------------------------------------------------------------------------------
    bool LogWriter::close_stream()
    {
        if (is_open())
        {
            _file.close();
            return true;
        }

        return false;
    }

    //---------------------------------------------------------------------------------------------
    bool LogWriter::write(const LogMessage *message)
    {
        if (_isMt)
        {
            // guard the CV
            std::lock_guard<std::mutex> guard(_cvLock);

            // push message in the queue ...
            const bool result = _messageQueue.push(message);
            
            // ... signal the thread that a new message is ready to be written, if we allow 100% cpu
            // set an event to true
            if (result)
                _cv.notify_one();

            return result;
        }
        else
        {
            // write without queueing
            const bool result = write_to_stream(message);
            delete message;
            return result;
        }
    }

    //---------------------------------------------------------------------------------------------
    bool LogWriter::is_open() const
    {
        return _file.is_open();
    }

    //---------------------------------------------------------------------------------------------
    bool LogWriter::is_mt() const
    {
        return _isMt;
    }
        
    //---------------------------------------------------------------------------------------------
    bool LogWriter::write_to_stream(const LogMessage *message)
    {
        assert((message != nullptr) && "LogWriter::write_to_stream : message is null!");
        if (is_open())
        {
            assert((_formatter != nullptr) && "LogWriter::write_to_stream : formatter is null!");
            _file << _formatter->get_message_header(message->get_subsystem(), message->get_type(), message->get_creation_time()) << message->get_streamer().str() << std::endl;
        }
        else
        {
            std::cout
                << message->get_subsystem() << "["
                << log_details::get_log_type_as_char(message->get_type()) << "]"
                << message->get_streamer().str()
                << std::endl;
        }
        return true;
    }

    //---------------------------------------------------------------------------------------------
    void LogWriter::run()
    {
        while (_run)
        {
            std::unique_lock<std::mutex> lock(_cvLock);
            while (_run && _messageQueue.empty()) // Guard for spurious wakeups
            {
                _cv.wait(lock);
            }

            const LogMessage *message;
            bool hasMessage = _messageQueue.pop(message);
            if (!hasMessage) continue;

            write_to_stream(message);
            delete message;
        }
    }

    //---------------------------------------------------------------------------------------------
    void LogWriter::start_thread()
    {
        _thread = std::thread(&LogWriter::run, this);
    }

    //---------------------------------------------------------------------------------------------
    void LogWriter::stop_thread()
    {
        if (_thread.joinable())
        {
            _run = false;
            _cv.notify_one(); // to release the thread from the _cv.wait.
            _thread.join();
        }
    }
} // namespace common
} // namespace gn