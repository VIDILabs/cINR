#include <GcCore/libCommon/Logger/LogCategory.hpp>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------
    LogCategory::LogCategory(const uint32_t id, const std::string &subsystem, const std::string &description,
        const log_details::Verbosity maxVerbosity)
    {
        _id = id;
        _subsystem = subsystem;
        _description = description;
        _maxVerbosity = maxVerbosity;
        _active = true;
    }

    //---------------------------------------------------------------------------------------------
    LogCategory::~LogCategory()
    {}

} // namespace common
} // namespace gn