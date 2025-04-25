#include <GcCore/libCommon/Logger/Logger.hpp>

#include <iostream>

#include <GcCore/libCommon/Logger/LogWriter.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------------
    Logger::Logger() : Singleton<Logger>()
    {
        _w = 8;
        _f = ' ';
        _writer = create_unique_ptr<LogWriter>();
    }

    //---------------------------------------------------------------------------------------------------
    Logger::~Logger()
    {}

    //---------------------------------------------------------------------------------------------------
    Logger& Logger::get_instance()
    {
        return Singleton<Logger>::get_instance();
    }

    //---------------------------------------------------------------------------------------------------
    bool Logger::open(const std::string &filePath, LoggerFormatter &formatter, bool isMT)
    {
        const bool result = _writer->open_stream(filePath, formatter, isMT);

        if (result)
        { 
            // @TODO : pass comand line args
            dump_cartridge(0, nullptr);
            dump_category();
        }
        
        return result;
    }

    //---------------------------------------------------------------------------------------------------
    bool Logger::close()
    {
        _writer.reset();
        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool Logger::add_category(const LogCategory &cat)
    {
        CategoryMapIterator it = _categories.find(cat.get_id());
        if (it != _categories.end())
        {
#if TDNS_MODE == TDNS_MODE_DEBUG
            std::cout << "Logger::add_category : Error : category id [" << cat.get_id() << "] already exists!" << std::endl;
#endif
            return false;
        }

        return _categories.insert(std::make_pair(cat.get_id(), cat)).second;
    }

    //---------------------------------------------------------------------------------------------------
    bool Logger::add_category(const uint32_t id, const std::string &subSystem, const std::string &desc,
        const log_details::Verbosity maxVerb)
    {
        return add_category({ id, subSystem, desc, maxVerb });
    }

    //---------------------------------------------------------------------------------------------------
    void Logger::log(const LogMessage *msg)
    {
        if (!msg)
        {
#if TDNS_MODE == TDNS_MODE_DEBUG
            std::cout << "Logger::log_message : Error : Message [nullptr]!" << std::endl;
#endif
            return;
        }

        if (_writer)
        {
            _writer->write(msg);
        }
#if TDNS_MODE == TDNS_MODE_DEBUG
        else
        {
            std::cout << "Logger::log_message : Error : no writed available!" << std::endl;
        }
#endif
    }

    //---------------------------------------------------------------------------------------------------
    bool Logger::find_category_and_check_validy(const uint32_t id, const log_details::Verbosity requested, LogCategory *&cat)
    {
        CategoryMapIterator it = _categories.find(id);
        if (it != _categories.end())
        {
            cat = &(it->second);

            if (it->second.is_active() && it->second.match_verbosity_level(requested))
            {
                return true;
            }
        }
        else
        {
            cat = nullptr;
        }

        return false;
    }

    //---------------------------------------------------------------------------------------------------
    void Logger::dump_category()
    {
        const char *tab[] = { "NONE", "LOW", "MEDIUM", "HIGH", "INSANE" };
        CategoryMapConstIterator it = _categories.begin();
        for (; it != _categories.end(); ++it)
        {
            LOGINFO(0, log_details::Verbosity::NONE, " + Category [" << it->second.get_id() << "] Short name [" << it->second.get_subsystem() << "]");
            LOGINFO(0, log_details::Verbosity::NONE, "    | Description [" << it->second.get_description() << "]");
            LOGINFO(0, log_details::Verbosity::NONE, "    ` Max verbosity [" << tab[it->second.get_max_verbosity()]<< "]");
        }
        LOGINFO(0, log_details::Verbosity::NONE, "**************************************************************************************************");
    }

    //---------------------------------------------------------------------------------------------------
    void Logger::dump_cartridge(int argc, char **argv)
    {
        std::stringstream stream;
        std::string prog_name;
        if (argc == 0)
        {
            stream << "Not given";
            prog_name = "Not given";
        }
        else
        {
            for (int i = 0; i < argc; ++i)
            {
                if (i == 0)
                {
                    prog_name = argv[i];
                }
                stream << argv[i] << ' ';
            }
        }

        LOGINFO(0, log_details::Verbosity::NONE, "**************************************************************************************************");
        LOGINFO(0, log_details::Verbosity::NONE, " Command line user : ");
        LOGINFO(0, log_details::Verbosity::NONE, stream.str());
        LOGINFO(0, log_details::Verbosity::NONE, "**************************************************************************************************");
    }

} // namespace common
} // namespace tdns