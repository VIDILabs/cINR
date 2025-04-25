#include <GcCore/libPython/Py_Logger.hpp>

#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/Logger/LoggerFormatterFileStd.hpp>

//---------------------------------------------------------------------------------------------------
void start_logger(char* path_name)
{
    static tdns::common::LoggerFormatterFileStd log_formatter;

    const tdns::common::LogCategory cat_default(0, "Default", "Default category to dump logs.", tdns::common::log_details::Verbosity::INSANE);
    const tdns::common::LogCategory cat_system(10, "System", "All logs related to the system. File access, etc.", tdns::common::log_details::Verbosity::INSANE);
    const tdns::common::LogCategory cat_Preprocessor(20, "Preprocessor", "All logs related to the preprocessing of the volumes", tdns::common::log_details::Verbosity::INSANE);
    const tdns::common::LogCategory cat_Graphics(30, "Graphics", "All logs related to the graphics part. SDL, window, etc.", tdns::common::log_details::Verbosity::INSANE);
    const tdns::common::LogCategory cat_GPUCache(40, "GPUCache", "All logs related to the cache system.", tdns::common::log_details::Verbosity::INSANE);

    tdns::common::Logger::get_instance().add_category(cat_default);
    tdns::common::Logger::get_instance().add_category(cat_system);
    tdns::common::Logger::get_instance().add_category(cat_Preprocessor);
    tdns::common::Logger::get_instance().add_category(cat_Graphics);
    tdns::common::Logger::get_instance().add_category(cat_GPUCache);

    tdns::common::Logger::get_instance().open(std::string(path_name), log_formatter, true);
}

//---------------------------------------------------------------------------------------------------
void end_logger()
{
    tdns::common::Logger::get_instance().close();
}