#pragma once

#include <cuda_runtime.h>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __host__ __device__ T create_default()
    {
        return T(0);
    }
    
    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ float create_default()
    {
        return 1.f;
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uchar1 create_default()
    {
        return make_uchar1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uchar2 create_default()
    {
        return make_uchar2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uchar3 create_default()
    {
        return make_uchar3(0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uchar4 create_default()
    {
        return make_uchar4(255, 0, 255, 255); // TMP : PINK !!
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ char1 create_default()
    {
        return make_char1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ char2 create_default()
    {
        return make_char2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ char3 create_default()
    {
        return make_char3(0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ char4 create_default()
    {
        return make_char4(0, 0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ ushort1 create_default()
    {
        return make_ushort1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ ushort2 create_default()
    {
        return make_ushort2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ ushort3 create_default()
    {
        return make_ushort3(0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ ushort4 create_default()
    {
        return make_ushort4(0, 0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ short1 create_default()
    {
        return make_short1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ short2 create_default()
    {
        return make_short2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ short3 create_default()
    {
        return make_short3(0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ short4 create_default()
    {
        return make_short4(0, 0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uint1 create_default()
    {
        return make_uint1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uint2 create_default()
    {
        return make_uint2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uint3 create_default()
    {
        return make_uint3(0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ uint4 create_default()
    {
        return make_uint4(0, 0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ int1 create_default()
    {
        return make_int1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ int2 create_default()
    {
        return make_int2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ int3 create_default()
    {
        return make_int3(0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ int4 create_default()
    {
        return make_int4(0, 0, 0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ ulonglong1 create_default()
    {
        return make_ulonglong1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ ulonglong2 create_default()
    {
        return make_ulonglong2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ longlong1 create_default()
    {
        return make_longlong1(0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ longlong2 create_default()
    {
        return make_longlong2(0, 0);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ float1 create_default()
    {
        return make_float1(0.f);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ float2 create_default()
    {
        return make_float2(0.f, 0.f);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ float3 create_default()
    {
        return make_float3(0.f, 0.f, 0.f);
    }

    //---------------------------------------------------------------------------------------------------
    template<>
    inline __host__ __device__ float4 create_default()
    {
        return make_float4(0.f, 0.f, 0.f, 0.f);
    }
} // namespace common
} // namespace tdns