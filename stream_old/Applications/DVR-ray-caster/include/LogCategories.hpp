#pragma once

#include <GcCore/libCommon/Logger/LogCategory.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>

namespace tdns
{
namespace app
{
    namespace LogCategories
    {
        const static tdns::common::LogCategory cat_default(0, "Default", "Default category to dump logs.", tdns::common::log_details::Verbosity::INSANE);
        const static tdns::common::LogCategory cat_system(10, "System", "All logs related to the system. File access, etc.", tdns::common::log_details::Verbosity::INSANE);
        const static tdns::common::LogCategory cat_Preprocessor(20, "Preprocessor", "All logs related to the preprocessing of the volumes", tdns::common::log_details::Verbosity::INSANE);
        const static tdns::common::LogCategory cat_Graphics(30, "Graphics", "All logs related to the graphics part. SDL, window, etc.", tdns::common::log_details::Verbosity::INSANE);
        const static tdns::common::LogCategory cat_GPUCache(40, "GPUCache", "All logs related to the cache system.", tdns::common::log_details::Verbosity::INSANE);

        void load_categories_in_logger()
        {
            tdns::common::Logger::get_instance().add_category(cat_default);
            tdns::common::Logger::get_instance().add_category(cat_system);
            tdns::common::Logger::get_instance().add_category(cat_Preprocessor);
            tdns::common::Logger::get_instance().add_category(cat_Graphics);
            tdns::common::Logger::get_instance().add_category(cat_GPUCache);
        }
    } // namespace LogCategories
} // namespace app
} // namespace tdns