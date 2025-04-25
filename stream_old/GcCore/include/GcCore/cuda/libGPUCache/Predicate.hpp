#pragma once

namespace tdns
{
namespace gpucache
{
namespace predicate
{

    /**
     * @brief
     */
    template<typename T>
    struct equal
    {
        equal(const T &ref)
        {
            _ref = ref;
        }
        /**
         * @brief
         */
        __host__ __device__
        inline bool operator()(const T &value)
        {
            return value == _ref;
        }

        T _ref;
    };

    /**
     * @brief
     */
    template<typename T>
    struct not_equal
    {
        not_equal(const T &ref)
        {
            _ref = ref;
        }
        /**
         * @brief
         */
        __host__ __device__
        inline bool operator()(const T &value)
        {
            return value != _ref;
        }

        T _ref;
    };

    template<typename T>
    struct is_greater_equal
    {
        is_greater_equal(const T &ref)
        {
            _ref = ref;
        }
        __host__ __device__
        inline bool operator()(const T &value)
        {
            return value >= _ref;
        }

        T _ref;
    };

} // namespace predicate
} // namespace gpucache
} // namespace tdns