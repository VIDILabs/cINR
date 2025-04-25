#include <GcCore/libData/Configuration.hpp>

#include <GcCore/libCommon/Logger/Logger.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    Configuration::Configuration()
    {}

    //---------------------------------------------------------------------------------------------------
    Configuration::~Configuration()
    {}

    //---------------------------------------------------------------------------------------------------
    Configuration& Configuration::get_instance()
    {
        return tdns::common::Singleton<Configuration>::get_instance();
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::load(std::function<void(Configuration&, const std::string&)> &parser, const std::string &file)
    {
        parser(*this, file);
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, const std::string &value)
    {
        if (key.empty())
        {
            LOGWARN(10, "Configuration: trying to add a field with an empty key.");
            return;
        }
        _values[key] = value;
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, uint32_t value)
    {
        add_field(key, std::to_string(value));
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, uint64_t value)
    {
        add_field(key, std::to_string(value));
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, int32_t value)
    {
        add_field(key, std::to_string(value));
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, int64_t value)
    {
        add_field(key, std::to_string(value));
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, float value)
    {
        add_field(key, std::to_string(value));
    }

    //---------------------------------------------------------------------------------------------------
    void Configuration::add_field(const std::string &key, double value)
    {
        add_field(key, std::to_string(value));
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, std::string &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;
        
        out = result->second;
        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, uint32_t &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;

        try
        {
            out = std::stoul(result->second);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, uint64_t &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;

        try
        {
            out = std::stoull(result->second);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, int32_t &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;

        try
        {
            out = std::stoi(result->second);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, int64_t &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;

        try
        {
            out = std::stoll(result->second);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, float &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;

        try
        {
            out = std::stof(result->second);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    //---------------------------------------------------------------------------------------------------
    bool Configuration::get_field(const std::string &key, double &out)  const
    {
        auto result = _values.find(key);
        if (result == _values.end()) return false;

        try
        {
            out = std::stod(result->second);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }
} //namespace data
} //namespace tdns