#include <thread>
#include <string>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/Logger/LoggerFormatterFileStd.hpp>

#include "LogCategories.hpp"
#include "Application.hpp"

int main(int argc, char **argv)
{
    tdns::common::LoggerFormatterFileStd log_formatter;
    try
    {
        //init the logger
        tdns::app::LogCategories::load_categories_in_logger();
        if (!tdns::common::is_dir("log")) tdns::common::create_folder("log");
        tdns::common::Logger::get_instance().open("./log/3dns", log_formatter, true);
        //start the app !
        std::string config = std::string(argv[1]);
        tdns::app::Application app(config);
        if (!app.init())
        {
            LOGFATAL(10, "Error while initializing the application.");
        }
        else
        {
            app.run();
        }
    }
    catch (const std::exception &ex)
    {
        LOGFATAL(0, "Fatal error: leaves the program with exception [" << ex.what() << "]");
        std::cout << ex.what() << std::endl;
    }

    LOGINFO(0, tdns::common::log_details::Verbosity::NONE, "Close the application.");
    //sleep to let the logger finishing to output the logs.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    tdns::common::Logger::get_instance().close();

//#if TDNS_OS == TDNS_OS_WINDOWS //&& TDNS_MODE == TDNS_MODE_DEBUG
    //system("pause");
//#endif
    return 0;
}
