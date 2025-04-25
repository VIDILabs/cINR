/*
 */
#pragma once

#include <cstdint>
#include <vector>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libMath/Vector.hpp>

namespace tdns
{
namespace data
{
    /**
    * @brief Brick class that give all data for a brick.
    */
    class TDNS_API Brick
    {
    public:

        /**
        * Default constructor
        */
        Brick(tdns::math::Vector3ui edgeSize = tdns::math::Vector3ui(32), uint8_t encoded = 1);

        /**
        * Constructor with a default value
        */
        Brick(uint8_t value, tdns::math::Vector3ui edgeSize, uint8_t encoded);

        /**
        * Constructor
        *
        * @param data the data vector corresponding of the data brick
        */
        Brick(const std::vector<uint8_t> &data, tdns::math::Vector3ui edgeSize = tdns::math::Vector3ui(32), uint8_t encoded = 1);

        /**
        * @brief Set all value of the brick to the given value.
        *
        * @param Value to set.
        */
        void fill(uint8_t value);
        
        /**
        * @brief Access operator overload.
        */
        uint8_t& operator [] (const uint32_t i);

        const uint8_t& operator [] (const uint32_t i) const;

        uint8_t& operator () (const uint32_t x, const uint32_t y, const uint32_t z);

        const uint8_t& operator () (const uint32_t x, const uint32_t y, const uint32_t z) const;

        uint8_t& operator () (const tdns::math::Vector3ui &position);

        const uint8_t& operator () (const tdns::math::Vector3ui &position) const;
        
        /**
        * Getters / Setters
        */
        std::vector<uint8_t>& get_data();

        const std::vector<uint8_t>& get_data() const;

        tdns::math::Vector3ui get_edge_size() const;

        uint32_t get_level() const;

        tdns::math::Vector3ui& get_position();

        const tdns::math::Vector3ui& get_position() const;

        uint32_t get_encoded() const;

        void set_data(const std::vector<uint8_t> &data);

        void set_data(const std::vector<uint8_t> &data, const uint32_t offset, const uint32_t size);

        void set_level(uint32_t level);

        void set_position(const tdns::math::Vector3ui &position);

    protected:

        using pad_t = char; ///< Used for padding.

        uint8_t                 _encoded;       ///< Number of bytes encoded for each voxel.
        pad_t                   _padding1[3];   ///< Memory padding.
        tdns::math::Vector3ui   _edgeSize;      ///< Edge size of a brick.
        uint32_t                _level;         ///< Precision level.
        tdns::math::Vector3ui   _position;      ///< Number of the brick on the 3 axis. (eg. 2nd(1) on X, 1st(0) on Y and 5th(4) on Z)
        std::vector<uint8_t>    _data;          ///< All data of the brick.
    };

    //---------------------------------------------------------------------------------------------------
    inline uint8_t& Brick::operator [] (const uint32_t i) { return _data[i]; }

    //---------------------------------------------------------------------------------------------------
    inline const uint8_t& Brick::operator [] (const uint32_t i) const { return _data[i]; }

    //---------------------------------------------------------------------------------------------------
    inline uint8_t& Brick::operator () (const uint32_t x, const uint32_t y, const uint32_t z)
    {
        return _data[(x + y * _edgeSize[0] + z * _edgeSize[1] * _edgeSize[2]) * _encoded];
    }

    //---------------------------------------------------------------------------------------------------
    inline const uint8_t& Brick::operator () (const uint32_t x, const uint32_t y, const uint32_t z) const
    {
        return _data[(x + y * _edgeSize[0] + z * _edgeSize[1] * _edgeSize[2]) * _encoded];
    }

    //---------------------------------------------------------------------------------------------------
    inline uint8_t& Brick::operator () (const tdns::math::Vector3ui &position)
    {
        return (*this)(position[0], position[1], position[2]);
    }

    //---------------------------------------------------------------------------------------------------
    inline const uint8_t& Brick::operator () (const tdns::math::Vector3ui &position) const
    {
        return (*this)(position[0], position[1], position[2]);
    }

    //---------------------------------------------------------------------------------------------------
    inline std::vector<uint8_t>& Brick::get_data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    inline const std::vector<uint8_t>& Brick::get_data() const
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    inline tdns::math::Vector3ui Brick::get_edge_size() const
    {
        return _edgeSize;
    }

    //---------------------------------------------------------------------------------------------------
    inline uint32_t Brick::get_level() const
    {
        return _level;
    }

    //---------------------------------------------------------------------------------------------------
    inline tdns::math::Vector3ui& Brick::get_position()
    {
        return _position;
    }

    //---------------------------------------------------------------------------------------------------
    inline const tdns::math::Vector3ui& Brick::get_position() const
    {
        return _position;
    }

    //---------------------------------------------------------------------------------------------------
    inline uint32_t Brick::get_encoded() const
    {
        return _encoded;
    }

    //---------------------------------------------------------------------------------------------------
    inline void Brick::set_data(const std::vector<uint8_t> &data)
    {
        std::copy(data.begin(), data.end(), _data.begin());
    }

    //---------------------------------------------------------------------------------------------------
    inline void Brick::set_data(const std::vector<uint8_t> &data, const uint32_t offset, const uint32_t size)
    {
        std::copy(data.begin() + offset, data.begin() + offset + size, _data.begin());
    }

    //---------------------------------------------------------------------------------------------------
    inline void Brick::set_level(uint32_t level)
    {
        _level = level;
    }

    //---------------------------------------------------------------------------------------------------
    inline void Brick::set_position(const tdns::math::Vector3ui &position)
    {
        _position = position;
    }
} //namespace data
} //namespace tdns
