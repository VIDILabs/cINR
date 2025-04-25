#pragma once

#include <cstdint>
#include <vector>
#include <thrust/device_vector.h>
#include <cuda.h>

#include <GcCore/libMath/Vector.hpp>

#include <GcCore/libCommon/NonCopyable.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/K_DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/StaticArray3dHost.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Class to create a 3D array on the device memory.
    *
    * @tparam [T]   Type of the data to store.
    */
    template<typename T>
    class DynamicArray3dDevice : public Noncopyable, KernelObject<K_DynamicArray3dDevice<T>>
    {
    public:
        /** Iterator to iterate over a DynamicArray3dDevice. */
        typedef typename thrust::device_vector<T>::iterator iterator;
        /** Constant iterator to iterate over a DynamicArray3dDevice. */
        typedef typename thrust::device_vector<T>::const_iterator const_iterator;

    public:

        /**
        * @brief Default constructor.
        */
        DynamicArray3dDevice() = default;

        /**
        * @brief Constructor with an array size.
        *
        * @param[in]    size    3D size of the array.
        */
        DynamicArray3dDevice(const tdns::math::Vector3ui &size);

        /**
        * @brief Constructor with an array size.
        *
        * @param[in]    size    3D size of the array.
        */
        DynamicArray3dDevice(const uint3 &size);

        /**
        * @brief Constructor with initialization of the array.
        *
        * @param[in]    size    3D size of the array.
        * @param[in]    value   Value to initialize the array.
        */
        DynamicArray3dDevice(const tdns::math::Vector3ui &size, const T &value);

        /**
        * @brief Constructor with initialization of the array.
        *
        * @param[in]    size    3D size of the array.
        * @param[in]    value   Value to initialize the array.
        */
        DynamicArray3dDevice(const uint3 &size, const T &value);
 
        /**
         * @brief Destructor.
         */
        virtual ~DynamicArray3dDevice();

        /**
        * @brief Operator overload.
        *
        * @param[in]    position    One or three dimensional position inside the data array.
        *
        * @return Read / Write iterator to the data at the given position.
        */
        //@{
        iterator operator [] (const size_t position);
        const_iterator operator [] (const size_t position) const;

        iterator operator () (const tdns::math::Vector3ui &position);
        const_iterator operator () (const tdns::math::Vector3ui &position) const;

        iterator operator () (const size_t x, const size_t y, const size_t z);
        const_iterator operator () (const size_t x, const size_t y, const size_t z) const;
        //@}

        /**
        * @brief Overload assigment operator.
        *
        * @paramparam[in]   data    An std::vector typed of T.
        */
        void operator = (const std::vector<T> &data);

        /**
        * @brief Get the begin iterator of the array.
        *
        * @return The begin iterator.
        */
        //@{
        iterator begin();
        const_iterator cbegin();
        //@}

        /**
        * @brief Get the end iterator of the array.
        *
        * @return The end iterator.
        */
        //@{
        iterator end();
        const_iterator cend();
        //@}

        /**
        * @brief Get the GPU pointer of the data array.
        * 
        * @return A pointer on data of the 3D GPU array.
        */
        //@{
        T* data();
        const T* data() const;
        //@}

        /**
        * @brief Get the one dimensional size of the array.
        *
        * @return The size.
        */
        size_t size() const;

        /**
        * @brief Get a CUDA Pitched memory pointer.
        * 
        * @return A CUDA Pitched memory pointer.
        */
        cudaPitchedPtr get_cuda_pitched_ptr();

        /**
        * @brief Get the object thrust::device_vector.
        *
        * @return The device vector.
        */
        //@{
        thrust::device_vector<T>& get_device_vector();
        const thrust::device_vector<T>& get_device_vector() const;
        //@}

        /**
        * @brief Get a kernem object that can be send to a CUDA kernel.
        *
        * @return A K_DynamicArray3dDevice object that can be send to a CUDA kernel.
        */
        virtual K_DynamicArray3dDevice<T> to_kernel_object() override;

    protected:
        /*
        * Member data.
        */
        uint32_t                    _xyProduct; ///< The product of the x and y size of the array.
        tdns::math::Vector3ui       _size;      ///< 3D size of the array.
        thrust::device_vector<T>    _data;      ///< Data of the 3D GPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    DynamicArray3dDevice<T>::DynamicArray3dDevice(const tdns::math::Vector3ui &size)
    {
        _xyProduct = size[0] * size[1];
        _size = size;

        _data.resize(size[0] * size[1] * size[2]);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    DynamicArray3dDevice<T>::DynamicArray3dDevice(const uint3 &size)
    {
        _xyProduct = size.x * size.y;

        _size = *reinterpret_cast<const tdns::math::Vector3ui*>(&size);
        
        _data.resize(size.x * size.y * size.z);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    DynamicArray3dDevice<T>::DynamicArray3dDevice(const tdns::math::Vector3ui &size, const T &value)
    {
        _xyProduct = size[0] * size[1];
        _size = size;

        _data.resize(size[0] * size[1] * size[2], value);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    DynamicArray3dDevice<T>::DynamicArray3dDevice(const uint3 &size, const T &value)
    {
        _xyProduct = size.x * size.y;

        _size = *reinterpret_cast<const tdns::math::Vector3ui*>(&size);

        _data.resize(size.x * size.y * size.z, value);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    DynamicArray3dDevice<T>::~DynamicArray3dDevice()
    {
        _data.clear();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::iterator DynamicArray3dDevice<T>::operator [] (const size_t position)
    {
        return _data.begin() + position;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::const_iterator DynamicArray3dDevice<T>::operator [] (const size_t position) const
    {
        return _data.begin() + position;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::iterator DynamicArray3dDevice<T>::operator () (const tdns::math::Vector3ui &position)
    {
        return _data.begin() + (position[0] + position[1] * _size[0] + position[2] * _xyProduct);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::const_iterator DynamicArray3dDevice<T>::operator () (const tdns::math::Vector3ui &position) const
    {
        return _data.begin() + (position[0] + position[1] * _size[0] + position[2] * _xyProduct);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::iterator DynamicArray3dDevice<T>::operator () (const size_t x, const size_t y, const size_t z)
    {
        return _data.begin() + (x + y * _size[0] + z * _xyProduct);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::const_iterator DynamicArray3dDevice<T>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        return _data.begin() + (x + y * _size[0] + z * _xyProduct);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void DynamicArray3dDevice<T>::operator = (const std::vector<T> &data)
    {
        _data = data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::iterator DynamicArray3dDevice<T>::begin()
    {
        return _data.begin();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::const_iterator DynamicArray3dDevice<T>::cbegin()
    {
        return _data.cbegin();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::iterator DynamicArray3dDevice<T>::end()
    {
        return _data.end();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline DynamicArray3dDevice<T>::const_iterator DynamicArray3dDevice<T>::cend()
    {
        return _data.cend();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline T* DynamicArray3dDevice<T>::data()
    {
        return _data.data().get();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const T* DynamicArray3dDevice<T>::data() const
    {
        return _data.data().get();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline size_t DynamicArray3dDevice<T>::size() const
    {
        return static_cast<size_t>(_size[0]) * static_cast<size_t>(_size[1]) * static_cast<size_t>(_size[2]);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline cudaPitchedPtr DynamicArray3dDevice<T>::get_cuda_pitched_ptr()
    {
        return make_cudaPitchedPtr(
            reinterpret_cast<void *> (_data.data().get()),
            static_cast<size_t>(_size[0]) * sizeof(T),
            static_cast<size_t>(_size[0]),
            static_cast<size_t>(_size[1]));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline thrust::device_vector<T>& DynamicArray3dDevice<T>::get_device_vector()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const thrust::device_vector<T>& DynamicArray3dDevice<T>::get_device_vector() const
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline K_DynamicArray3dDevice<T> DynamicArray3dDevice<T>::to_kernel_object()
    {
        return K_DynamicArray3dDevice<T>(_data.data().get(), _size);
    }
} // namespace tdns
} // namespace common

/**
* @class tdns::common::DynamicArray3dDevice
* @ingroup gpucache
*
* Create a 3D array on the device memory typed of T.
* Because it uses a thrust::device_vector<T>, it is possible to get
* the data stored on the device using accessor overload [] or ().
* 
* Example:
* @code
* tdns::math::Vector3ui array_size(10, 20, 30);
* tdns::common::DynamicArray3dDevice<uint32_t> array(array_size);
* array[0] = 42; //same as array(tdns::math::Vector3ui(0, 0, 0)) or array(0, 0, 0)
* @endcode
*/