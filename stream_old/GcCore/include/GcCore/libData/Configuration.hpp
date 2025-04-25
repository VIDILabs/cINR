#pragma once

#include <string>
#include <cstdint>
#include <functional>
#include <map>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Singleton.hpp>
#include <GcCore/libData/ConfigurationPolicies.hpp>
namespace tdns
{
namespace data
{
    /**
    * @brief
    */
    class TDNS_API Configuration : public tdns::common::Singleton<Configuration>
    {
    public:

        /**
        * @brief Default constructor
        */
        Configuration();

        /**
        * @brief Destructor
        */
        ~Configuration();

        static Configuration& get_instance();

        /**
        * @brief Load the a configuration file.
        * 
        * Allows to load multiple configuration file but if two keys are identical
        * the last inserted will overwrite the existing one.
        *
        * @template Parser used to load the configuration file.
        * @param Path to the configuration file.
        */
        void load(std::function<void(Configuration&, const std::string&)> &parser, const std::string &file);

        template<typename T>
        void load(const std::string &file)
        {
            T::load_from_file(*this, file);
        }

        /**
        * Add a pair field - value in the configuration.
        * /!\ Be carefull, if the field already exists it will overwrite it.
        * 
        * @param Name of the field.
        * @param The corresponding value to store.
        */
        void add_field(const std::string &key, const std::string &value);
        void add_field(const std::string &key, uint32_t value);
        void add_field(const std::string &key, uint64_t value);
        void add_field(const std::string &key, int32_t value);
        void add_field(const std::string &key, int64_t value);
        void add_field(const std::string &key, float value);
        void add_field(const std::string &key, double value);

        /**
        * @brief Get the value of a given field.
        *
        * @param Name of the field.
        * @param The value of the field.
        *
        * @return True if the output value has been successfully set, false otherwise.
        */
        bool get_field(const std::string &key, std::string &out) const;
        bool get_field(const std::string &key, uint32_t &out) const;
        bool get_field(const std::string &key, uint64_t &out) const;
        bool get_field(const std::string &key, int32_t &out) const;
        bool get_field(const std::string &key, int64_t &out) const;
        bool get_field(const std::string &key, float &out) const;
        bool get_field(const std::string &key, double &out) const;

    protected:
        /**
        * Member data
        */
        std::map<std::string, std::string> _values;
    };


} //namespace data
} //namespace tdns