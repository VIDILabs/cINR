#pragma once

#include <utility>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief CRTP Based singleton implementation.
    *
    * This class provides basic stuff to handle singleton.
    * This implementation is not thread-safe.
    * @tparam [T] The class / type to instantiate as a singleton.
    *          Nothing special required regarding its interface: default constructor if you
    *          intend to create the singleton without parameters or a valid variadic constructor
    *          if parameters are given.
    */
    template <typename T>
    class TDNS_API Singleton
    {
    public:

        /**
        * @brief Default constructor (do nothing).
        */
        Singleton();

        /**
        * @brief Default destructor (do nothing).
        *
        * It deletes the instance not the object. For a full destruction use detroy().
        */
        virtual ~Singleton();

        /**
        * @brief Construct the underlying object.
        *
        * Create the T object by calling its the default constructor.
        * @return A valide pointer to the object T.
        */
        //static T* get_instance();
        static T& get_instance();

        /**
        * @brief Construct the underlying object with a variadic parameters list.
        *
        * Create the T object by calling the most appropriate constructor regarding
        * the variadic parameters list.
        * @return A valide pointer to the object T.
        */
        template <typename... Args>
        //static T* get_instance(Args... args);
        static T& get_instance(Args... args);

        /**
        * @brief Frees the memory taken by the singleton object.
        *
        * This methode has no effect if the no calls to get_instance has been made.
        */
        static void destroy();

    private:

        static T* _pT; ///< instance of the T object.
    };

    //---------------------------------------------------------------------------------------------
    template <typename T>
    T* Singleton<T>::_pT = nullptr;

    //---------------------------------------------------------------------------------------------
    template <typename T>
    inline Singleton<T>::Singleton() {}

    //---------------------------------------------------------------------------------------------
    template <typename T>
    inline Singleton<T>::~Singleton() {}

    //---------------------------------------------------------------------------------------------
    template <typename T>
    inline T& Singleton<T>::get_instance()
    {
        if (!_pT)
        {
            _pT = new T();
        }
        return *_pT;
    }

    //---------------------------------------------------------------------------------------------
    template <typename T>
    template <typename... Args>
    inline T& Singleton<T>::get_instance(Args... args)
    {
        if (!_pT)
        {
            _pT = new T(std::forward<Args>(args)...);
        }
        return *_pT;
    }

    //---------------------------------------------------------------------------------------------
    template <typename T>
    inline void Singleton<T>::destroy()
    {
        if (_pT)
        {
            delete _pT;
            _pT = nullptr;
        }
    }
} // namespace common
} // namespace tdns