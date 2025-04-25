/*
 */
#include <GcCore/libData/Brick.hpp>

#include <cstring>

#include <GcCore/libCommon/Logger/Logger.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    Brick::Brick(tdns::math::Vector3ui edgeSize /* = tdns::math::Vector3ui(32) */, uint8_t encoded /* = 1 */)
    {
        _edgeSize = edgeSize;
        _level = 0;
        _position = tdns::math::Vector3ui(0);
        _encoded = encoded;
        _data.resize(encoded * edgeSize[0] * edgeSize[1] * edgeSize[2]);
        std::fill(_data.begin(), _data.end(), 0);
    }

    //---------------------------------------------------------------------------------------------------
    Brick::Brick(uint8_t value, tdns::math::Vector3ui edgeSize, uint8_t encoded)
    {
        _edgeSize = edgeSize;
        _level = 0;
        _position = tdns::math::Vector3ui(0);
        _encoded = encoded;
        _data.resize(encoded * edgeSize[0] * edgeSize[1] * edgeSize[2]);
        std::fill(_data.begin(), _data.end(), value);
    }

    //---------------------------------------------------------------------------------------------------
    Brick::Brick(const std::vector<uint8_t> &data, tdns::math::Vector3ui edgeSize /* = tdns::math::Vector3ui(32) */, uint8_t encoded /* = 1 */)
    {
        _edgeSize = edgeSize;
        _level = 0;
        _position = tdns::math::Vector3ui(0);
        _encoded = encoded;

        size_t dataSize = encoded * edgeSize[0] * edgeSize[1] * edgeSize[2];
        if (dataSize != data.size())
        {
            LOGERROR(10, "Unable to create a brick from given data. Data size do not match. Brick data size [" << dataSize 
                << "] - data size [" << data .size() << "].");
            std::fill(_data.begin(), _data.end(), 0);
            return;
        }

        _data.resize(dataSize);
        std::copy(data.begin(), data.end(), _data.begin());
    }

    //---------------------------------------------------------------------------------------------------
    void Brick::fill(uint8_t value)
    {
        std::fill(_data.begin(), _data.end(), value);
    }
} //namespace data
} //namespace tdns