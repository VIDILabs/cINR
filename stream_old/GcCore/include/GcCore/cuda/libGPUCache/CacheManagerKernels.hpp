#pragma once

#include <cuda_runtime.h>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/cuda/libCommon/K_DynamicArray3dDevice.hpp>

#include <GcCore/cuda/libGPUCache/LruContent.hpp>
#include <GcCore/cuda/libGPUCache/K_CacheManager.hpp>
#include <GcCore/cuda/libGPUCache/K_MultiResolutionPageDirectory.hpp>
#include <GcCore/cuda/libGPUCache/K_CacheDevice.hpp>

#include "../../../../../ovr/api.h"

namespace tdns
{
namespace gpucache
{
    //---------------------------------------------------------------------------------------------------
    /**
    * @brief Fill a boolean array given a timestamp and the a usage buffer.
    *
    * @param blockSize [description]
    * @param lru       [description]
    * @param usage     [description]
    * @param timestamp [description]
    * @param mask      [description]
    */
    __global__ void TDNS_API create_usage_mask(uint3 blockSize,
        LruContent* __restrict__ lru,
        size_t lruSize,
        tdns::common::K_DynamicArray3dDevice<uint32_t> usage,
        uint32_t timestamp,
        bool *mask);

    //---------------------------------------------------------------------------------------------------
    __global__ void TDNS_API coord_1D_to_3D(size_t *indexes,
        size_t nbElements,
        uint4* __restrict__ positions,
        tdns::common::K_DynamicArray3dDevice<uint3> realNumberOfEntries);

    //---------------------------------------------------------------------------------------------------
    __device__ void TDNS_API compute_volume_position(LruContent *content,
        uint4 position,
        uint3 size,
        float3 one_over_brickSize,
        float epsilon);

    //---------------------------------------------------------------------------------------------------
    __global__ void TDNS_API  compute_lru_volume_position(LruContent *lruDraft,
        tdns::common::K_DynamicArray3dDevice<uint3> volumeSizes,
        uint4 *requestedBricks,
        uint32_t nbBricks,
        float3 one_over_brickSize,
        float epsilon);

    //---------------------------------------------------------------------------------------------------
    __global__ void TDNS_API copy_lru_position(LruContent *lruTo, LruContent *lruFrom, size_t size);

    //---------------------------------------------------------------------------------------------------
    __global__ void fillAllCoords(const uint3 brick_size, uint4* __restrict__ requested_bricks, const vnr::vec3f vol_dims, const uint32_t count,
                                  vnr::vec3f* __restrict__ d_coords, vnr::vec3f* __restrict__ d_outCoords);

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    static __global__ void write_brick(K_CacheDevice<T> dataCache,
        T* __restrict__ bricks,
        LruContent *position,
        uint4* __restrict__ requestedBricks,
        uint32_t *bigBrickIndexes,
        uint4 *bigBrickCoords,
        size_t nbBricks,
        uint3 brickSize,
        uint3 bigBrickSize,
        uint3 bigBrickVoxels)
    {
        uint3 voxelPos;
        voxelPos.x = (blockIdx.x * blockDim.x) + threadIdx.x;
        voxelPos.y = (blockIdx.y * blockDim.y) + threadIdx.y;
        voxelPos.z = (blockIdx.z * blockDim.z) + threadIdx.z;

        if (voxelPos.x >= brickSize.x || voxelPos.y >= brickSize.y || voxelPos.z >= brickSize.z) return;

        size_t offsetBigBrickVoxel = bigBrickVoxels.x * bigBrickVoxels.y * bigBrickVoxels.z;

        //position of the voxel in the brick from the beginin of the small brick
        int32_t positionInBrick = (voxelPos.z * bigBrickVoxels.y + voxelPos.y) * bigBrickVoxels.x + voxelPos.x;

        for (uint32_t n = 0; n < nbBricks; ++n)
        {
            //position of the big brick in the buffer
            size_t bigBrickPositionInBuffer = bigBrickIndexes[n] * offsetBigBrickVoxel;

            //delta position from the begining of the big brick and the begining of the small brick
            uint4 SB_coord = requestedBricks[n];
            uint4 BB_coord = bigBrickCoords[bigBrickIndexes[n]];
            size_t x = SB_coord.x - BB_coord.x * bigBrickSize.x;
            size_t y = SB_coord.y - BB_coord.y * bigBrickSize.y;
            size_t z = SB_coord.z - BB_coord.z * bigBrickSize.z;
            size_t smallBrickPosition = ((z * bigBrickVoxels.y + y) * bigBrickVoxels.x + x) * brickSize.x;
            
            //absolute position of the voxel in the buffer
            uint32_t absolutePositionInBuffer = bigBrickPositionInBuffer + smallBrickPosition + positionInBrick;

            uint3 cachePosition = (position + n)->cachePosition;

            cachePosition.x += voxelPos.x;
            cachePosition.y += voxelPos.y;
            cachePosition.z += voxelPos.z;

            dataCache.insert(cachePosition, bricks[absolutePositionInBuffer]);
        }
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void TDNS_API cache_use(LruContent *lru, size_t size, uint32_t *count);

    // Dereferenced the old brick that will be remove from the data cache.
    // We dereference the entry of the last PTcache in the hierarchy, or the MRPD if not.
    //---------------------------------------------------------------------------------------------------
    template<typename T>
    static __global__ void dereference_old_bricks(K_CacheManager<T> manager, LruContent *positions, uint32_t nbBricks)
    {
        uint32_t threadX = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadX >= nbBricks) return;

        float3 position = positions[threadX].volumePosition;
        if (position.x < 0.f) return; //replace a non mapped brick.(empty place in the LRU)

        uint32_t level = positions[threadX].level;

        const K_MultiResolutionPageDirectory<uint4> &mrpd = manager.get_mrpd();

        uint32_t nbTableCaches = manager.get_nb_table_caches();

        if (nbTableCaches >= 1)
        {
            uint4 entry = mrpd.get(level, position);
            K_CacheDevice<uint4> *tableCaches = manager.get_table_caches();
            for (uint32_t i = 0; i < nbTableCaches - 1; ++i)
                entry = tableCaches[i].get(entry, level, position);

            tableCaches[nbTableCaches - 1].insert(entry, level, position, make_uint4(0, 0, 0, 0));
        }
        else // MRPD only
            mrpd.set(level, position, make_uint4(0, 0, 0, 0));

        // For morpho math : reset the entries corresponding to all the voxels of the removed brick in the flag buffer to 0
        manager.reset_data_cache_buffer_entries(level, position);

        // printf("derefPos=(%f, %f, %f) | %u\n", position.x, position.y, position.z, threadX);
    }

    //---------------------------------------------------------------------------------------------------
    inline __device__ void fill_stack(LruContent *stack, size_t size, LruContent **lrus, uint32_t *lruSizes, size_t *indexLRU)
    {
        for (size_t i = 0; i < size; ++i)
        {
            stack[i] = lrus[i][(lruSizes[i] - 1) - indexLRU[i]];
            ++indexLRU[i];
        }
    }

    //---------------------------------------------------------------------------------------------------
    /**
    * @brief Compute the callstack from the MRPD to the data cache for a given brick.
    *        It will say from the MRPD to which PT cache the brick is referenced.
    *        It means after the last PT cache, PT blocks will be required to reference the new brick.
    *        In the stack, if:
    *           * volumePosition == -2.f => nothing to add a this level.
    *           * volumePosition == -1.f => new block to add, nothing to remove.
    *           * volumePosition > -1.F  => old block to remove, new block to add.
    */
    inline __device__ void downward(const LruContent &brick,
        LruContent *stack,
        const K_MultiResolutionPageDirectory<uint4> &mrpd,
        K_CacheDevice<uint4> *tableCaches,
        uint32_t nbTableCaches,
        LruContent **lrus,
        uint32_t *lruSizes,
        size_t *indexLRU)
    {
        uint3 cachePosition = brick.cachePosition;
        const float3 &position = brick.volumePosition;
        uint32_t level = brick.level;

        stack[nbTableCaches + 1].cachePosition = cachePosition;
        stack[nbTableCaches + 1].level = level;
        stack[nbTableCaches + 1].volumePosition = position;

        // printf("Brick position : %f, %f, %f - level %d\n", position.x, position.y, position.z, level);
        // printf("Cache position : %d, %d, %d - level %d\n", cachePosition.x, cachePosition.y, cachePosition.z, level);

        uint4 entry = mrpd.get(level, position); //looking for the entry of the new brick

        int32_t i = 0;
        while (true)     // going down into the hierarchy to notice the first unmapped entry
        {
            if (entry.w == 1)
            {
                if (i >= nbTableCaches) { break; } // TODO not sure if this is the correct fix
                // printf("MAPPED\n");
                stack[i].cachePosition = make_uint3(entry.x, entry.y, entry.z);
                stack[i].level = level;
                stack[i].volumePosition = make_float3(-2.f, -2.f, -2.f);
                entry = tableCaches[i].get(entry, level, position);
                ++i;
            }
            else
            {
                // printf("PAS MAPPED\n");
                // if at least one PT cache : FILL THE STACK with the last element of each cache LRU
                if (nbTableCaches != 0)
                {
                    fill_stack(
                        &stack[i == 0 ? 1 : i],
                        i == 0 ? nbTableCaches : nbTableCaches - i + 1,
                        &lrus[i == 0 ? 0 : i - 1],
                        lruSizes,
                        indexLRU);
                }

                break;
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    inline __device__ void upward(const LruContent &brick,
        LruContent *stack,
        const K_MultiResolutionPageDirectory<uint4> &mrpd,
        K_CacheDevice<uint4> *tableCaches,
        uint32_t nbTableCaches,
        LruContent **lrus,
        uint32_t *lruSizes,
        size_t *indexLRU,
        bool empty = false)
    {
        const float3 &position = brick.volumePosition;
        uint32_t level = brick.level;

        int32_t i = nbTableCaches; //go to the last cache. Or 0 if only the MRPD
        uint4 entry;

        while (true)          // going up to update the hierarchy
        {
            if (i < 0 || stack[i].volumePosition.x < -1.f) break; // end of loop

            if (i == 0)//==> mrpd
            {
                entry = mrpd.get(level, position);
                if (entry.w >= 1) break; //end of update - already mapped or empty
                const uint3 &cacheEntry = stack[i + 1].cachePosition;
                mrpd.set(level, position, make_uint4(
                    cacheEntry.x,
                    cacheEntry.y,
                    cacheEntry.z,
                    empty ? 2 : 1));
            }
            else
            {
                //check if we need to add a new entry or complete an existing one
                if (stack[i - 1].volumePosition.x < -1.f) //complete an existing one
                {
                    const uint3 &cacheEntry = stack[i - 1].cachePosition;
                    //flag the new entry
                    tableCaches[i - 1].insert(
                        make_uint4(cacheEntry.x, cacheEntry.y, cacheEntry.z, 0),
                        level,
                        position,
                        make_uint4(stack[i + 1].cachePosition.x, stack[i + 1].cachePosition.y, stack[i + 1].cachePosition.z, 1));
                    break;
                }

                // Entry in wich we will insert the PT of the new brick
                const uint3 &cacheEntry = stack[i].cachePosition;

                // reset entry in the current cache
                tableCaches[i - 1].reset_entries(cacheEntry);

                if (stack[i].volumePosition.x > -1.f)
                {
                    //unflag from the previous cache. We must go back from the top level (the MRPD)
                    uint4 tmpEntry = mrpd.get(stack[i].level, stack[i].volumePosition);
                    if (i == 1)
                        mrpd.set(stack[i].level, stack[i].volumePosition, make_uint4(0, 0, 0, 0));
                    else
                    {
                        for (uint32_t j = 0; j < i - 1; ++j)
                            tmpEntry = tableCaches[j].get(tmpEntry, stack[i].level, stack[i].volumePosition);

                        tableCaches[i - 2].insert(
                            tmpEntry,
                            stack[i].level,
                            stack[i].volumePosition,
                            make_uint4(0, 0, 0, 0));
                    }
                }

                //flag the new entry
                tableCaches[i - 1].insert(
                    make_uint4(cacheEntry.x, cacheEntry.y, cacheEntry.z, 0),
                    level,
                    position,
                    make_uint4(stack[i + 1].cachePosition.x, stack[i + 1].cachePosition.y, stack[i + 1].cachePosition.z, 1));

                // LRU update
                lrus[i - 1][(lruSizes[i - 1] - 1) - indexLRU[i - 1] + 1].volumePosition = position;
                lrus[i - 1][(lruSizes[i - 1] - 1) - indexLRU[i - 1] + 1].level = level;
            }
            i--;
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    static __global__ void reference_new_bricks(K_CacheManager<T> manager,
        LruContent *newPositions,
        LruContent **lrus,
        LruContent *emptyBricksPositions,
        uint32_t *lruSizes,
        uint32_t nbEmptyBricks,
        uint32_t nbNonEmptyBricks)
    {
        // if kernel called with <<<a,b>>> not <<<1,1>>>, will issue illegal instruction
        if ((blockDim.x * blockDim.y * blockDim.z != 1) || (gridDim.x * gridDim.y * gridDim.z != 1))
            asm("trap;");

        const K_MultiResolutionPageDirectory<uint4> &mrpd = manager.get_mrpd();
        const uint32_t nbTableCaches = manager.get_nb_table_caches();
        K_CacheDevice<uint4> *tableCaches = manager.get_table_caches();

        LruContent *stack = new LruContent[nbTableCaches + 2];
        memset(stack, 0, (nbTableCaches + 2) * sizeof(LruContent));

        size_t *indexLRU = nullptr;
        if (nbTableCaches != 0)
        {
            indexLRU = new size_t[nbTableCaches];
            memset(indexLRU, 0, nbTableCaches * sizeof(size_t));
        }

        // non empty bricks referencing
        for (uint32_t brickIndex = 0; brickIndex < nbNonEmptyBricks; ++brickIndex)
        {
            //going downward to look up to where the brick is referenced.
            downward(newPositions[brickIndex], stack, mrpd, tableCaches, nbTableCaches, lrus, lruSizes, indexLRU);
            
            //going upward to reference the brick.
            upward(newPositions[brickIndex], stack, mrpd, tableCaches, nbTableCaches, lrus, lruSizes, indexLRU);
        }

        // empty bricks referencing
        for (uint32_t brickIndex = 0; brickIndex < nbEmptyBricks; ++brickIndex)
        {
            //going downward to look up to where the brick is referenced.
            downward(emptyBricksPositions[brickIndex], stack, mrpd, tableCaches, nbTableCaches, lrus, lruSizes, indexLRU);

            //going upward to reference the brick.
            upward(emptyBricksPositions[brickIndex], stack, mrpd, tableCaches, nbTableCaches, lrus, lruSizes, indexLRU, true);
        }

        delete stack;
        if (indexLRU) delete indexLRU;
    }
} // namespace gpucache
} // namespace tdns
