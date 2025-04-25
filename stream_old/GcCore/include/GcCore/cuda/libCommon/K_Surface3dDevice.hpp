#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace tdns
{
namespace common
{
    /**
    * @brief Kernel class to call in a kernel to use a surface.
    *
    * @template Type of the data stored in the surface.
    */
    template<typename T>
    class K_Surface3dDevice
    {
    public:

        /**
        * @brief Constructor.
        *
        * @param The surface this object will refere to.
        * It does not become the owner of it.
        */
        K_Surface3dDevice(cudaSurfaceObject_t surface);

        /**
        * @brief Device getter on the surface memory.
        * 
        * @param The 3D position inside the surface.
        * 
        * @return The data at the given 3D position inside the surface.
        */
        __device__ T get(const uint3 &position) const;

        /**
        * @brief Device setter on the surface memory.
        * 
        * @param The 3D position inside the surface.
        * @param The value to set.
        */
        __device__ void set(const uint3 &position, const T &value) const;

    protected:
        /**
        * Member data.
        */
        cudaSurfaceObject_t _surface;   ///< The surface object.
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    K_Surface3dDevice<T>::K_Surface3dDevice(cudaSurfaceObject_t surface)
    {
        _surface = surface;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T K_Surface3dDevice<T>::get(const uint3 &position) const
    {
        // CMAKE ERROR with the templated version of surf3Dread()
#if TDNS_OS == TDNS_OS_LINUX
        T data;
        surf3Dread(&data, _surface, position.x * sizeof(T), position.y, position.z);
        return data;
#else
        return surf3Dread<T>(_surface, position.x * sizeof(T), position.y, position.z);
#endif
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    void K_Surface3dDevice<T>::set(const uint3 &position, const T &value) const
    {
        surf3Dwrite(value, _surface, position.x * sizeof(T), position.y, position.z);
    }
} // namespace tdns
} // namespace common