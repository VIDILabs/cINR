#pragma once

#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/MPMCBoundedQueue.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Forward declarations.
    */
    class LogMessage;
    class LoggerFormatter;

    /**
    * @brief As its name says, it writes in the log file all logs.
    *        The class can be use with a thread.
    */
    class TDNS_API LogWriter
    {
    public:
        /**
        * @brief Default constructor.
        */
        LogWriter();

        /**
        * @brief Desctructor.
        */
        ~LogWriter();

        /**
        * @brief Open the log file.
        * Appends the date at the end of the file name and puts the extension to '.log'.
        * The file is created only, it must not exists.
        * @return true on success, false if :
        *           - the log is already opened
        *           - if the file already exists
        *           - the file could not be created
        */
        bool open_stream(const std::string &filePath, LoggerFormatter &formatter, bool isMt);

        /**
        * @brief Close the log file.
        *
        * Close the log file.
        * @return true on success, false otherwise.
        */
        bool close_stream();

        /**
        * @brief Write the log message.
        *
        * If the log is NOT multithreaded it will directly write the log in the log file.
        * Otherwise it will queue the message and then notify the logging thread a 
        * new message has been queued.
        *
        * @param Log message to dump.
        */
        bool write(const LogMessage *message);

        /**
        * @brief Check if the log file is opened.
        * 
        * Indicate if the log file is opened.
        * @return True if the log fileis opened, false otherwise.
        */
        bool is_open() const;

        /**
        * @brief Check if the writer works on the main thread or on its own thead.
        *
        * @return True if it run on its own thread, false otherwaise.
        */
        bool is_mt() const;

    private:

        bool write_to_stream(const LogMessage *message);

        void run();

        void start_thread();

        void stop_thread();

    private:

        bool                                _isMt;          ///< Boolean to set the logger on a thread or not.
        std::ofstream                       _file;          ///< File where the log will be dumped.
        MPMCBoundedQueue<const LogMessage*> _messageQueue;  ///< Queue used to store the logs if the MT flag is true.
        LoggerFormatter                     *_formatter;    ///< Provide the string format the logs will follow.

        std::thread                         _thread;        ///< Thread use to dequeue if the MT flag is true;
        std::mutex                          _cvLock;        ///< Mutex only used to notify the thread a new log is queued.
        std::condition_variable             _cv;            ///< Condition variable the thread will listen in order to dequeue.
        bool                                _run;           ///< Boolean stop the logger.
    };
} // namespace common
} // namespace gn