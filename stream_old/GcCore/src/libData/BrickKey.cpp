#include <GcCore/libData/BrickKey.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    Bkey get_key(const std::string &strKey)
    {
        assert(false && "get_key(const std::string &strKey) not implemented yet.");
        return 0;
    }

    //---------------------------------------------------------------------------------------------------
    Bkey get_key(uint32_t level, const tdns::math::Vector3ui &position)
    {
        uint16_t buffer[4];

        buffer[0] = static_cast<uint16_t>(level);
        buffer[1] = static_cast<uint16_t>(position[0]);
        buffer[2] = static_cast<uint16_t>(position[1]);
        buffer[3] = static_cast<uint16_t>(position[2]);

        return *(reinterpret_cast<Bkey*>(buffer));
    }

    //---------------------------------------------------------------------------------------------------
    std::string get_brick_name(Bkey key)
    {
        uint16_t *array = reinterpret_cast<uint16_t*>(&key);
        return "L" + std::to_string(array[0]) + "_" +
            std::to_string(array[1]) + "_" +
            std::to_string(array[2]) + "_" +
            std::to_string(array[3]);
    }

    //---------------------------------------------------------------------------------------------------
    std::string get_brick_name(uint32_t level, const tdns::math::Vector3ui &position)
    {
        return "L" + std::to_string(level) + "_" +
            std::to_string(position[0]) + "_" +
            std::to_string(position[1]) + "_" +
            std::to_string(position[2]);
    }

    //---------------------------------------------------------------------------------------------------
    void get_brick_level_position(Bkey key, uint32_t &level, tdns::math::Vector3ui &position)
    {
        uint16_t *array = reinterpret_cast<uint16_t*>(&key);
        level = static_cast<uint32_t>(array[0]);
        position[0] = static_cast<uint32_t>(array[1]);
        position[1] = static_cast<uint32_t>(array[2]);
        position[2] = static_cast<uint32_t>(array[3]);
    }

    //---------------------------------------------------------------------------------------------------
    void get_brick_level_position(const std::string &strKey, uint32_t &level, tdns::math::Vector3ui &position)
    {
        assert(false && 
            "get_brick_level_position(const std::string &strKey, uint32_t &level, tdns::math::Vector3ui &position) not implemented yet.");
    }
} //namespace data
} //namespace tdns