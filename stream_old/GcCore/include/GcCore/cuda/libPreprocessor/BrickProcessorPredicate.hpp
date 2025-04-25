#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace tdns
{
namespace preprocessor
{
    struct DefaultBrickProcessorPredicate
    {
        /**
        * @brief Default predicate. If the value of the current voxel is
        *        under a threshold, it is considered as an empty voxel.
        */
        __device__ static bool predicate(uchar1 voxel, void *otherData)
        {
            return voxel.x < *reinterpret_cast<uint8_t*>(otherData) ? true : false;
        }
    };

    struct BrickProcessor16BitsPredicate
    {
        /**
        * @brief Default predicate. If the value of the current voxel is
        *        under a threshold, it is considered as an empty voxel.
        */
        __device__ static bool predicate(ushort1 voxel, void *otherData)
        {
            return voxel.x < *reinterpret_cast<uint16_t*>(otherData) ? true : false;
        }
    };

    struct BrickProcessor32BitsPredicate
    {
        /**
        * @brief Default predicate. If the value of the current voxel is
        *        under a threshold, it is considered as an empty voxel.
        */
        __device__ static bool predicate(uint1 voxel, void *otherData)
        {
            return voxel.x < *reinterpret_cast<uint32_t*>(otherData) ? true : false;
        }
    };

    struct BrickProcessorFloatPredicate
    {
        /**
        * @brief Default predicate. If the value of the current voxel is
        *        under a threshold, it is considered as an empty voxel.
        */
        __device__ static bool predicate(float1 voxel, void *otherData)
        {
            return voxel.x < *reinterpret_cast<float*>(otherData) ? true : false;
        }
    };

    struct ProcessorUchar4Predicate
    {
        /**
        * @brief RGBA (uchar4) volume predicate. If the RGB value of the current voxel is
        *       comprise in the range (determine by a threshold) of one of the RGB
        *       color listed, it is considered as an empty voxel.
        */
        __device__ static bool predicate(uchar4 voxel, void *otherData)
        {
            int8_t *ptr = reinterpret_cast<int8_t*>(otherData);
            size_t threshold = *reinterpret_cast<size_t*>(ptr);
            ptr += sizeof(size_t);
            size_t size = *reinterpret_cast<size_t*>(ptr);
            ptr += sizeof(size_t);
            uchar3 *colors = *reinterpret_cast<uchar3**>(ptr);

            for (size_t i = 0; i < size; ++i)
            {
                if (  (voxel.x >= static_cast<int32_t>(colors[i].x - threshold)) && (voxel.x <= colors[i].x + threshold)
                    &&(voxel.y >= static_cast<int32_t>(colors[i].y - threshold)) && (voxel.y <= colors[i].y + threshold)
                    &&(voxel.z >= static_cast<int32_t>(colors[i].z - threshold)) && (voxel.z <= colors[i].z + threshold)) return true;
            }

            return false;
        }
    };

    template<typename Type>
    struct BrickProcessorGenericPredicate_1
    {
        /**
        * @brief Default predicate. If the value of the current voxel is
        *        under a threshold, it is considered as an empty voxel.
        */
        __device__ static bool predicate(Type voxel, void *otherData)
        {
            return voxel.x < *reinterpret_cast<decltype(voxel.x)*>(otherData) ? true : false;
        }
    };

} // namespace preprocessor
} // namespace tdns