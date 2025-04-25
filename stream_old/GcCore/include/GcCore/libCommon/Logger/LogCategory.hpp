#pragma once

#include <string>
#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Logger/LogCommon.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Basic object that ease the categories manipulation.
    *
    * This object is used internally by the logger itself in order to manipulate and manage
    * the different categories that are added in a program.
    * A category is made of 4 information:
    *  - A numeric id. It is used every time you need to log an information.
    *  - A subsytem. It is a short string used to indicate in the header the subsystem that
    *    emitted the log.
    *  - A description. Usually dump after the cartridge, it gives somes details about the
    *    category purpose.
    *  - A maximum verbosity. All logging tentatives with a verbosity greater than the
    *    provided one will be dropped.
    */
    class TDNS_API LogCategory
    {
    public:

        /**
        * @brief Disable default constructor.
        */
        LogCategory() = delete;

        /**
        * @brief Classic constructor.
        * 
        * @param[in]    id             The numerical unique id identifies this category amongst others.
        * @param[in]    subsystem      Short string, human readable, that represents the system that has emitted the log.
        * @param[in]    description    String that explains the category's purpose.
        * @param[in]    maxVerbosity   Maximum verbosity allowed to log on this category.
        */
        LogCategory(const uint32_t id, const std::string &subsystem, const std::string &description,
            const log_details::Verbosity maxVerbosity);

        /**
        * @brief Destructor.
        */
        ~LogCategory();
            
        /**
        * @brief Getter for the category's id.
        *
        * @return The id.
        */
        uint32_t get_id() const;

        /**
        * @brief Getter for the category's subsystem.
        *
        * @return The subsytem.
        */
        const std::string& get_subsystem() const;

        /**
        * @brief Getter for the category's description.
        *
        * @return The description.
        */
        const std::string& get_description() const;

        /**
        * @brief Getter for the category's maximum verbosity.
        *
        * @return The verbosity level.
        */
        tdns::common::log_details::Verbosity get_max_verbosity() const;

        /**
        * @brief Check if the given verbosity match with the verbosity allowed.
        *
        * @param[in]    verbosity  The verbosity to test.
        *
        * @return True if the provided verbosity allows a dump, false otherwise.
        */
        bool match_verbosity_level(const tdns::common::log_details::Verbosity verbosity) const;

        /**
        * @brief Change the maximum verbosity level.
        *
        * @param[in]    verbosity  The new maximum verbosity.
        */
        void change_max_verbosity(const log_details::Verbosity verbosity);

        /**
        * @brief Indicates if the category is active and can be used for a dump.
        *
        * @return True if active, false otherwise.
        */
        bool is_active() const;

        /**
        * @brief Turn On/Off the category.
        *
        * @param[in]    status     Status of the category. Set to true to switch on, false to switch off.
        * @note By default, every category is active.
        */
        void set_active(const bool status);

    protected:
        uint32_t                _id;            ///< Numeric id. Must be unique.
        std::string             _subsystem;     ///< String that identifies the subsystem.
        std::string             _description;   ///< String that provides a brief information about the category.
        log_details::Verbosity  _maxVerbosity;  ///< Maximum verbosity allowed for the category.
        bool                    _active;        ///< Boolean to know if the category is active or not.
    };

    //---------------------------------------------------------------------------------------------
    inline uint32_t LogCategory::get_id() const
    {
        return _id;
    }

    //---------------------------------------------------------------------------------------------
    inline const std::string& LogCategory::get_subsystem() const
    {
        return _subsystem;
    }

    //---------------------------------------------------------------------------------------------
    inline const std::string& LogCategory::get_description() const
    {
        return _description;
    }

    //---------------------------------------------------------------------------------------------
    inline log_details::Verbosity LogCategory::get_max_verbosity() const
    {
        return _maxVerbosity;
    }

    //---------------------------------------------------------------------------------------------
    inline bool LogCategory::match_verbosity_level(const log_details::Verbosity verbosity) const
    {
        return _maxVerbosity >= verbosity;
    }

    //---------------------------------------------------------------------------------------------
    inline void LogCategory::change_max_verbosity(const log_details::Verbosity verbosity)
    {
        _maxVerbosity = verbosity;
    }

    //---------------------------------------------------------------------------------------------
    inline bool LogCategory::is_active() const
    {
        return _active;
    }

    //---------------------------------------------------------------------------------------------
    inline void LogCategory::set_active(const bool status)
    {
        _active = status;
    }
} // namespace common
} // namespace tdns