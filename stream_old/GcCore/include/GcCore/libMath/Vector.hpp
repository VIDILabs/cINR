#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <array>

#include <GcCore/libMath/MathUtils.hpp>

namespace tdns
{
namespace math
{
    /**
    * @brief Template class to create vector size of 2, 3, 4.
    *
    * @tparam [T]   The type of the values in the vector.
    * @tparam [N]   The size of the vector.
    * @tparam [ulp] (Units in the Last Place) see: MathUtils.hpp
    */
    template <typename T, int n, uint32_t ulp = 4>
    class Vector
    {
    public:
        /**
        * @brief Default constructor.
        */
        Vector() { _v.fill(0); }

        /**
        * @brief Constructor with a default value.
        *
        * @param value  Value to set.
        */
        Vector(const T &value) { _v.fill(value); }

        /**
        * @brief Constructor with values for a Vector2.
        *
        * @param x  Value of the first element.
        * @param y  Value of the second element.
        */
        Vector(const T &x, const T &y)
        {
            static_assert(n == 2, "Bad vector size - Wrong type used for this number of parameter");
            this->_v[0] = x;
            this->_v[1] = y;
        }

        /**
        * @brief Constructor with values for a Vector3.
        *
        * @param x  Value of the first element.
        * @param y  Value of the second element.
        * @param z  Value of the third element.
        */
        Vector(const T &x, const T &y, const T &z)
        {
            static_assert(n == 3, "Bad vector size - Wrong type used for this number of parameter");
            this->_v[0] = x;
            this->_v[1] = y;
            this->_v[2] = z;
        }

        /**
        * @brief Constructor with values for a Vector4.
        *
        * @param x  Value of the first element.
        * @param y  Value of the second element.
        * @param z  Value of the third element.
        * @param w  Value of the fourth element.
        */
        Vector(const T &x, const T &y, const T &z, const T &w)
        {
            static_assert(n == 4, "Bad vector size - Wrong type used for this number of parameter");
            this->_v[0] = x;
            this->_v[1] = y;
            this->_v[2] = z;
            this->_v[3] = w;
        }

        /**
        * @brief Constructor from a different size of vector.
        * It creates a vector of size n and fill it with the values
        * of the vector of size m. If n < m the vector is truncated
        * and if n > m, extendedValue is added to the end.
        *
        * @param vec            Vector that need to be copied.
        * @param extendedValue  Extended value(s) if m < n.
        */
        template<int m>
        explicit Vector(const Vector<T, m> &vec, const T &extendedValue = T(0))
        {
            for (int32_t i = 0; i < std::min(m, n); ++i) this->_v[i] = vec[i];
            for (int32_t i = std::min(m, n); i < n; ++i) this->_v[i] = extendedValue;
        }

        /**
        * @brief Access operator overload.
        */
        T& operator [] (const int i) { return this->_v[i]; }

        const T& operator [] (const int i) const { return this->_v[i]; }

        T& operator () (const int i) { return this->_v[i]; }

        const T& operator () (const int i) const { return this->_v[i]; }

        /**
        * @brief Overload of mathematics operator.
        */
        Vector& operator -= (const Vector &other)
        {
            for (uint32_t i = 0; i < n; ++i) this->_v[i] -= other._v[i];
            return *this;
        }

        Vector& operator += (const Vector &other)
        {
            for (uint32_t i = 0; i < n; ++i) this->_v[i] += other._v[i];
            return *this;
        }

        Vector& operator *= (const T &value)
        {
            for (uint32_t i = 0; i < n; ++i) this->_v[i] *= value;
            return *this;
        }

        Vector& operator *= (const Vector &other)
        {
            for (uint32_t i = 0; i < n; ++i) this->_v[i] *= other._v[i];
            return *this;
        }

        Vector& operator /= (const T &value)
        {
            for (uint32_t i = 0; i < n; ++i) this->_v[i] /= value;
            return *this;
        }

        Vector& operator /= (const Vector &other)
        {
            for (uint32_t i = 0; i < n; ++i) this->_v[i] /= other._v[i];
            return *this;
        }

        Vector operator + (const Vector &other) const { return Vector(*this) += other; }

        Vector operator - () const { return Vector(*this) *= -1; }

        Vector operator - (const Vector &other) const { return Vector(*this) -= other; }

        Vector operator * (const T &value) const { return Vector(*this) *= value; }

        Vector operator * (const Vector &other) const { return Vector(*this) *= other; }

        Vector operator / (const T &value) const { return Vector(*this) /= value; }
        
        Vector operator / (const Vector &other) const { return Vector(*this) /= other; }

        /**
        * @brief Overload equals operator (for double and float).
        */
        bool operator == (const Vector &other) const
        {
            for (uint32_t i = 0; i < n; ++i)
            {
                if (!tdns::math::utils::almost_equal(this->_v[i], other._v[i], ulp)) return false;
            }
            return true;
        }

        bool operator != (const Vector &other) const { return !(*this == other); }
        
        /**
        * @brief Get direct access to the vector data.
        *
        * @return A pointer to the vector data.
        */
        T* data()
        {
            return _v.data();
        }

        /**
        * @brief Get direct access to the vector data.
        *
        * @return A constant pointer to the vector data.
        */
        const T* data() const
        {
            return _v.data();
        }

        /**
        * @brief Give the norm of the vector.
        */
        T get_norm() const { return std::sqrt(this->dot_product(*this)); }

        /**
        * @brief Normalize the vector.
        */
        Vector& normalize()
        {
            assert(this->dot_product(*this) > std::numeric_limits<T>::epsilon());
            return *this /= this->get_norm();
        }

        /**
        * @brief Give the normalize vector of the vector.
        *
        * @return Normalize vector.
        */
        Vector get_normalize_vector() const
        {
            assert(this->dot_product(*this) > std::numeric_limits<T>::epsilon());
            return*this / this->get_norm();
        }

        /**
        * @brief Dot product between this vector and 
        * the given vector.
        *
        * @param A vector.
        *
        * @return Dot product between both vector.
        */
        T dot_product(const Vector &other) const
        {
            T result = 0;
            for (uint32_t i = 0; i < n; ++i) result += this->_v[i] * other[i];
            return result;
        }

        /**
        * @brief Give the cross product between this vector and 
        * the given vector.
        *
        * @params A vector.
        *
        * @return The cross product vector.
        */
        Vector<T, 3> cross_product(const Vector<T, 3> &other) const
        {
            static_assert(n == 3, "Bad vector size - Wrong type used for *this");
            return Vector<T, 3>(this->_v[1] * other[2] - this->_v[2] * other[1], //multiplied by +1
                this->_v[2] * other[0] - this->_v[0] * other[2], //multiplied by -1
                this->_v[0] * other[1] - this->_v[1] * other[0]);//multiplied by +1
        }

    private:
        /*
        * Member data
        */
        std::array<T, n> _v; ///< data 
    }; //End class Vector

    //Element of type double precision float
    typedef Vector <double, 2> Vector2;
    typedef Vector <double, 3> Vector3;
    typedef Vector <double, 4> Vector4;

    //Element of type single precision float
    typedef Vector <float, 2> Vector2f;
    typedef Vector <float, 3> Vector3f;
    typedef Vector <float, 4> Vector4f;

    //Element of type unsigned byte
    typedef Vector <uint8_t, 2> Vector2b;
    typedef Vector <uint8_t, 3> Vector3b;
    typedef Vector <uint8_t, 4> Vector4b;

    //Element of type unsigned int32
    typedef Vector <uint32_t, 2> Vector2ui;
    typedef Vector <uint32_t, 3> Vector3ui;
    typedef Vector <uint32_t, 4> Vector4ui;

    //Element of type signed int32
    typedef Vector <int32_t, 2> Vector2i;
    typedef Vector <int32_t, 3> Vector3i;
    typedef Vector <int32_t, 4> Vector4i;
} //namespace math
} //namespace tdns

/**
* @class tdns::math::Vector
* @ingroup math
* 
* IT eases the use of a mathematics vector and provides
* basic vector operations.
* The default values of a vector are 0.
*
* Example:
* @code
* tdns::math::Vector3ui vec1(1), vec2(2); //Create vector of 3 * uint32_t
* tnds::math::Vector3ui vec3 = vec1 + vec2;
* (vec3 == tnds::math::Vector3ui(3, 3, 3)) // TRUE
* @endcode
*/