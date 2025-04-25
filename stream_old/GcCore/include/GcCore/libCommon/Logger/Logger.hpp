#pragma once

#include <iostream>
#include <cstdint>
#include <map>
#include <memory>

#include <GcCore/libCommon/Singleton.hpp>
#include <GcCore/libCommon/Logger/LogCategory.hpp>
#include <GcCore/libCommon/Logger/LogMessage.hpp>
#include <GcCore/libCommon/Logger/LoggerFormatter.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    class LogWriter;
    /**
    @brief A logger with MT capabilities.
    *
    * This logger provides a category mechanism to manage the data to dump.
    * Also, the logging subsystem can work in the main thread or using
    * multi-threading thechnics.
    */
    class TDNS_API Logger : public Singleton<Logger>
    {
    public:
        enum Category
        {
            CAT_KNL = 0
        };

    protected:
        using CategoryMap = std::map<uint32_t, LogCategory>;            ///< Type that associates id with a category
        using CategoryMapIterator = CategoryMap::iterator;              ///< Type for iterating the category container
        using CategoryMapConstIterator = CategoryMap::const_iterator;   ///< Type for iterating the category container

    public:
        /**
        * @brief Default constructor.
        *
        * Set up the logger subsystem with default values;
        * - single thread
        * - no log file
        * - no category set
        */
        Logger();

        /**
        * @brief Destrcutor.
        *
        * Clean the log subsystem.
        */
        ~Logger();

        static Logger& get_instance();

        /**
        * @brief Open a new stream for log
        *
        * @param Prefix of the file path to which the log will be redirected.
        * @param Line formatter for this stream.
        * @param Flag to indicate if we want the stream to be multithread,
        *           ie: if we can push message from multiple thread to a fix hardware thread
        * @param Affinity ?
        *
        * @return true if succes, false otherwise.
        */
        bool open(const std::string &filePath, LoggerFormatter &formatter, bool isMT = false);

        /**
        * @brief Close the log file.
        *
        * Close the log file.
        * @return true on success, false otherwise.
        */
        bool close();

        /**
        * @brief Add a log category.
        *
        * Reference a log category within the logger.
        * @param Th category to reference.
        * 
        * @return false if the category already exists, true otherwise.
        */
        bool add_category(const LogCategory &cat);

        /**
        * @brief Add a log category.
        *
        * Reference a log category within the logger with the given information.
        * @param The numerical id that uniquely identifies this category amongst the others.
        * @param Short string, human readable, that represents the system that has emitted the log.
        * @param A string that explains the category's purpose.
        * @param Maxomum verbosity allowed for loggin on this category.
        *
        * @return false if the category already exists, true otherwise.
        */
        bool add_category(const uint32_t id, const std::string &subSystem, const std::string &desc,
            const log_details::Verbosity maxVerb);

        /**
        * @brief Send a log message to the log writer.
        */
        void log(const LogMessage *msg);

        /**
        * Check if the message's attributes given are valid to send the message to the writer.
        */
        bool find_category_and_check_validy(const uint32_t id, const log_details::Verbosity requested, LogCategory *&cat);

        /**
        * @brief Dumpas all categories referenced in the logger.
        * 
        * Dumps all the categories referenced in the logger. 
        * These information are dumped just after the cartridge.
        */
        void dump_category();

        /**
        * @brief Dump a cartridge with basic informations.
        *
        * Dump a cartridge with basic information about the program and the host it is running on.
        * @param Number of arguements on the command line.
        * @param List of arguments on the line.
        */
        void dump_cartridge(int argc, char **argv);

    protected:
        CategoryMap                 _categories;    ///< The categories.
        std::streamsize             _w;             ///< Used for filling when dealing with floating points.
        char                        _f;             ///< Used for filling when dealing with floating points.
        std::unique_ptr<LogWriter>  _writer;        ///< responsible of log writing.
    };
} // namespace common
} // namespace tdns

#if TDNS_MODE == TDNS_MODE_RELEASE
#   define LOGMESSAGE(CATEGORY, TYPE, VERBOSITY, MSG)                                                   \
    tdns::common::LogCategory *cat = nullptr;                                                           \
    if (tdns::common::Logger::get_instance().find_category_and_check_validy(CATEGORY, VERBOSITY, cat))  \
    {                                                                                                   \
        if(cat)                                                                                         \
        {                                                                                               \
            tdns::common::LogMessage *msg = new tdns::common::LogMessage(cat, TYPE, VERBOSITY);         \
            msg->get_streamer() << MSG;                                                                 \
            tdns::common::Logger::get_instance().log(msg);                                              \
        }                                                                                               \
    }
#else
#   define LOGMESSAGE(CATEGORY, TYPE, VERBOSITY, MSG)                                                                                       \
    tdns::common::LogCategory *cat = nullptr;                                                                                               \
    if (tdns::common::Logger::get_instance().find_category_and_check_validy(CATEGORY, VERBOSITY, cat))                                      \
    {                                                                                                                                       \
        if(cat)                                                                                                                             \
        {                                                                                                                                   \
            tdns::common::LogMessage *msg = new tdns::common::LogMessage(cat, TYPE, VERBOSITY);                                             \
            msg->get_streamer() << MSG;                                                                                                     \
            tdns::common::Logger::get_instance().log(msg);                                                                                  \
        }                                                                                                                                   \
        else                                                                                                                                \
        {                                                                                                                                   \
            tdns::common::LogMessage *msg = new tdns::common::LogMessage();                                                                 \
            msg->get_streamer() << MSG;                                                                                                     \
            std::cout << "LOGMESSAGE : Error : Message [" << msg->get_streamer().str() << "] discarded no category found!" << std::endl;    \
        }                                                                                                                                   \
    }                                                                                                                                       \
    else                                                                                                                                    \
    {                                                                                                                                       \
        tdns::common::LogMessage *msg = new tdns::common::LogMessage();                                                                     \
        msg->get_streamer() << MSG;                                                                                                         \
        std::cout << "LOGMESSAGE : Error : Message [" << msg->get_streamer().str() << "] discarded!" << std::endl;                          \
    }
#endif

#define LOGTRACE(CAT, VERBOSITY, MSG) do { LOGMESSAGE(CAT, tdns::common::log_details::LogType::TRACE, VERBOSITY, MSG); } while(false)
#define LOGINFO(CAT, VERBOSITY, MSG) do { LOGMESSAGE(CAT, tdns::common::log_details::LogType::INFO, VERBOSITY, MSG); } while(false)
#define LOGWARN(CAT, MSG) do { LOGMESSAGE(CAT, tdns::common::log_details::LogType::WARN, tdns::common::log_details::Verbosity::NONE, MSG); } while(false)
#define LOGERROR(CAT, MSG) do { LOGMESSAGE(CAT, tdns::common::log_details::LogType::ERROR_TEST, tdns::common::log_details::Verbosity::NONE, MSG); } while(false)
#define LOGFATAL(CAT, MSG) do { LOGMESSAGE(CAT, tdns::common::log_details::LogType::FATAL_TEST, tdns::common::log_details::Verbosity::NONE, MSG); } while(false)

#if TDNS_MODE == TDNS_MODE_DEBUG
#   define LOGDEBUG(CATEGORY, VERBOSITY, MSG) do { LOGMESSAGE(CATEGORY, tdns::common::log_details::LogType::DEBUG, VERBOSITY, MSG); } while(false)
#else
#   define LOGDEBUG(CATEGORY, VERBOSITY, MSG) do {} while(false)
#endif