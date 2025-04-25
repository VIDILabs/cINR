#pragma once

#include <array>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libMath/Vector.hpp>

namespace tdns
{
namespace math
{
    /**
    * @brief a quad is a 
    * v0 _____ v1
    *  |       |
    *  |       |
    * v3 _____ v2
    */
    class TDNS_API Quad
    {
    public:
        /**
        * @brief Constructor
        */
        Quad();

        /**
        * @brief Access operator overload.
        */
        Vector3& operator [] (const int i) { return this->_v[i]; }

        const Vector3& operator [] (const int i) const { return this->_v[i]; }

        Vector3& operator () (const int i) { return this->_v[i]; }

        const Vector3& operator () (const int i) const { return this->_v[i]; }
    private:
        /**
        * Member data
        */
        std::array<Vector3, 4> _v;
    };
} //namespace math
} //namespace tdns