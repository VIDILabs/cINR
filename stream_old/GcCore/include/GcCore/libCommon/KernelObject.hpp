#pragma once

namespace tdns
{
namespace common
{
    /**
    * @brief Abstract class to create a kernel object.
    *        It is an object that can be passed as parameter
    *        in a CUDA kernel.
    *
    * @tparam [T]   Type of the kernel object that will be created.
    */
    template<typename T>
    class KernelObject
    {
    public:

        /**
        * @brief Default constructor.
        */
        KernelObject() = default;

        /**
        * @brief Destructor.
        */
        virtual ~KernelObject() = default;

        /**
        * @brief Pure virtual method the will need to be overridden.
        *        It will create the kernel object that can be used in
        *        a kernel.
        * 
        * @return Kernel object given in template parameter.
        */
        virtual T to_kernel_object() = 0;
    };
} // namespace tdns
} // namespace common