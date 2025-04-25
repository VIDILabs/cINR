#pragma once

#include <cassert>

namespace tdns
{
namespace common
{
    namespace log_details
    {
        /**
        * @brief Type to deal with the verbosity level of the logs.
        *
        */
        typedef enum
        {
            NONE,       ///< Lowest level possible. Only log with this verbosity level will be dumped.
            LOW,        ///< Provides some information about subsytem, without entering into the deeps.
            MEDIUM,     ///< Starts to enter into details.
            HIGH,       ///< Even more details.
            INSANE      ///< Consider it only for debugging. Verbosity level is at its maximum.
        } Verbosity;

        /**
        * @brief Type to deal with the type of the logs.
        */
        typedef enum
        {
            DEBUG,
            TRACE,
            INFO,
            WARN,
            ERROR_TEST,
            FATAL_TEST
        } LogType;

        inline char get_log_type_as_char(LogType type)
        {
            assert((type >= LogType::DEBUG && type <= LogType::FATAL_TEST) && "get_log_type_as_char : argument out of range.");
            static char tab[] = { 'D', 'T', 'I', 'W', 'E', 'F' };
            return tab[type];
        }
    } // namespace log_details
} // namespace common
} // namespace tdns