#pragma once

#include <cstdint>
#include <string>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libMath/Vector.hpp>

namespace tdns
{
namespace data
{
    typedef uint64_t Bkey;    ///< id created to identify a brick.


    /**
    * @brief Create the brick key (id) from the brick name.
    *
    * The brick name norm is: LN_X_Y_Z with,
    *       - N = level of precision (0 = max precision).
    *       - X = Brick numbore on X axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Y = Brick numbore on Y axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Z = Brick numbore on Z axis (0 means the first brick, 1 means the second brick, etc.).
    *   eg. L0_1_0_2
    *
    * @param[in]    Brick name.
    *
    * @return       Brick key.
    */
    TDNS_API Bkey get_key(const std::string &strKey);

    /**
    * @brief Create the brick key (id) from the brick name.
    *
    * The brick name norm is: LN_X_Y_Z with,
    *       - N = level of precision (0 = max precision).
    *       - X = Brick numbore on X axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Y = Brick numbore on Y axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Z = Brick numbore on Z axis (0 means the first brick, 1 means the second brick, etc.).
    *   eg. L0_1_0_2
    *
    * @param[in]    Precision level of the brick.
    * @param[in]    Position of the brick.
    *
    * @return       Brick key.
    */
    TDNS_API Bkey get_key(uint32_t level, const tdns::math::Vector3ui &position);

    /**
    * @brief Create the normalize brick name from the unique id of the brick.
    *
    * The brick name norm is: LN_X_Y_Z with,
    *       - N = level of precision (0 = max precision).
    *       - X = Brick numbore on X axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Y = Brick numbore on Y axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Z = Brick numbore on Z axis (0 means the first brick, 1 means the second brick, etc.).
    *   eg. L0_1_0_2
    *
    * @param[in]    Brick key.
    *
    * @return       The normalize brick name.
    */
    TDNS_API std::string get_brick_name(Bkey key);

    /**
    * @brief Create the normalize brick name from the level and the brick position.
    *
    * The brick name norm is: LN_X_Y_Z with,
    *       - N = level of precision (0 = max precision).
    *       - X = Brick numbore on X axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Y = Brick numbore on Y axis (0 means the first brick, 1 means the second brick, etc.).
    *       - Z = Brick numbore on Z axis (0 means the first brick, 1 means the second brick, etc.).
    *   eg. L0_1_0_2
    *
    * @param[in]    Precision level of the brick.
    * @param[in]    Position of the brick.
    *
    * @return       The normalize brick name.
    */
    TDNS_API std::string get_brick_name(uint32_t level, const tdns::math::Vector3ui &position);

    TDNS_API void get_brick_level_position(Bkey key, uint32_t &level, tdns::math::Vector3ui &position);
    TDNS_API void get_brick_level_position(const std::string &strKey, uint32_t &level, tdns::math::Vector3ui &position);
} //namespace data
} //namespace tdns