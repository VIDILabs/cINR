#pragma once
 
#include <cstdint>
#include <limits>
#include <type_traits>
#include <cmath>

namespace tdns
{
namespace math
{
    namespace utils
    {
        /**
        * @brief Function taken from cppreference in order to 
        * compare two different double or float.*
        *
        * @param x      First double / float.
        * @param y      Second double / float.
        * @param ulp    (Units in the Last Place) The larger the value is, 
        *               the more error we allow. 0 means two values must be
        *               exactly the same.
        *
        *  @see     
        *       - http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        *       - http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
        *       - http://stackoverflow.com/questions/13698927/compare-double-to-zero-using-epsilon
        *       - http://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison
        */
        template<typename T>
        typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, uint32_t ulp = 4)
        {
            // required because if we are in the case we want to compare x = 0, y = 1e-17 
            // we can consider y as 0 but the almost_equals won't works.
            // And 1e-10 is clearly enough to be sure that a value == 0.
            if(std::abs(x + y) < 1.0) return std::abs(x - y) < 1e-10;
 
            // the machine epsilon has to be scaled to the magnitude of the values used
            // and multiplied by the desired precision in ULPs (units in the last place)
            return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
                // unless the result is subnormal
                || std::abs(x - y) < std::numeric_limits<T>::min();
        }


        /**
        * @brief Normalize a value from an interval to another interval.
        *
        * @param[in]    originMin   Minimum value of the origin interval.
        * @param[in]    originMax   Maximum value of the origin interval.
        * @param[in]    input       Value to transform.
        * @param[in]    newMin      Minimum value of the new interval.
        * @param[in]    newMax      Maximum value of the new interval.
        *
        * @return Normalize value in the new interval.
        */
        template<typename T>
        T normalize(T originMin, T originMax, T input, T newMin, T newMax)
        {
            return ((input - originMin) / (originMax - originMin)) * (newMax - newMin) + newMin;
        }
    } //namespace utils
} //namespace math
} //namespace tdns