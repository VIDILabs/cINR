#include <GcCore/cuda/libGPUCache/CacheManagerKernels.hpp>

namespace tdns
{
namespace gpucache
{
    //---------------------------------------------------------------------------------------------------
    __global__ void create_usage_mask(uint3 blockSize,
        LruContent* __restrict__ lru,
        size_t lruSize,
        tdns::common::K_DynamicArray3dDevice<uint32_t> usage,
        uint32_t timestamp,
        bool *mask)
    {

        // index in the LRU
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= lruSize) return;

        // position in the usage buffer
        uint3 position = lru[index].cachePosition;

        position.x /= blockSize.x;
        position.y /= blockSize.y;
        position.z /= blockSize.z;

        // printf("Cache pos:(%u, %u, %u) | Vol pos: (%f, %f, %f)\n", position.x, position.y, position.z, lru[index].volumePosition.x, lru[index].volumePosition.y, lru[index].volumePosition.z);

        mask[index] = (usage(position) == timestamp);
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void coord_1D_to_3D(size_t *indexes,
        size_t nbElements,
        uint4* __restrict__ positions,
        tdns::common::K_DynamicArray3dDevice<uint3> realNumberOfEntries)
    {
        uint32_t thIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (thIndex >= nbElements) return;

        uint3 size = make_uint3(0, 0, 0);

        for (size_t i = 0; i < realNumberOfEntries.size(); ++i)
            size.x += realNumberOfEntries[i].x;
        size.y = realNumberOfEntries[0].y;
        size.z = realNumberOfEntries[0].z;

        size_t index = indexes[thIndex];

        uint3 position;

        position.z = static_cast<uint32_t>(index / (size.x * size.y));
        index -= position.z * (size.x * size.y);
        position.y = static_cast<uint32_t>(index / size.x);
        index -= position.y * size.x;
        position.x = static_cast<uint32_t>(index);

        uint32_t sumX = 0;
        size_t level;

        for (level = 0; level < realNumberOfEntries.size(); ++level)
        {
            sumX += realNumberOfEntries[level].x;
            if (position.x < sumX)
                break;
        }

        positions[thIndex].x = position.x - (sumX - realNumberOfEntries[level].x);
        positions[thIndex].y = position.y;
        positions[thIndex].z = position.z;
        positions[thIndex].w = level;
    }

    //---------------------------------------------------------------------------------------------------
    __device__ void compute_volume_position(LruContent *content,
        uint4 position,
        uint3 size,
        float3 one_over_brickSize,
        float epsilon)
    {
        content->volumePosition.x = static_cast<float>(position.x) / (static_cast<float>(size.x) * one_over_brickSize.x) + epsilon;
        content->volumePosition.y = static_cast<float>(position.y) / (static_cast<float>(size.y) * one_over_brickSize.y) + epsilon;
        content->volumePosition.z = static_cast<float>(position.z) / (static_cast<float>(size.z) * one_over_brickSize.z) + epsilon;

        content->level = position.w;
    }

    // Compute the volume position in the LRU for each new bricks
    //---------------------------------------------------------------------------------------------------
    __global__ void compute_lru_volume_position(LruContent *lruDraft,
        tdns::common::K_DynamicArray3dDevice<uint3> volumeSizes,
        uint4 *requestedBricks,
        uint32_t nbBricks,
        float3 one_over_brickSize,
        float epsilon)
    {
        uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadId >= nbBricks) return;

        uint32_t level = requestedBricks[threadId].w;

        compute_volume_position(&lruDraft[threadId], requestedBricks[threadId], volumeSizes[level], one_over_brickSize, epsilon);
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void copy_lru_position(LruContent *lruTo, LruContent *lruFrom, size_t size)
    {
        uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= size) return;

        lruTo[threadId].level = lruFrom[threadId].level;
        lruTo[threadId].volumePosition = lruFrom[threadId].volumePosition;
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void fillAllCoords(const uint3 brick_size, uint4* __restrict__ requested_bricks, const vnr::vec3f vol_dims, const uint32_t count,
                                  vnr::vec3f* __restrict__ d_coords, vnr::vec3f* __restrict__ d_outCoords) {
    
        vnr::vec3i id(threadIdx.x + blockIdx.x * blockDim.x,
                      threadIdx.y + blockIdx.y * blockDim.y,
                      threadIdx.z + blockIdx.z * blockDim.z);

        // Calculate brick index and coordinates within the brick
        const uint32_t brick_id = id.x / brick_size.x;
        id.x %= brick_size.x;
        id.y %= brick_size.y;
        id.z %= brick_size.z;

        if (brick_id >= count)    return;
        if (id.x >= brick_size.x) return;
        if (id.y >= brick_size.y) return;
        if (id.z >= brick_size.z) return;

        const vnr::vec3i brick_offset(requested_bricks[brick_id].x * (brick_size.x - 2),
                                      requested_bricks[brick_id].y * (brick_size.y - 2),
                                      requested_bricks[brick_id].z * (brick_size.z - 2));
        const uint32_t lod_scale = pow(2,requested_bricks[brick_id].w);

        // Calculate the linear index for the specific coordinate
        const uint32_t idx = (id.z * brick_size.x * brick_size.y) + (id.y * brick_size.x) + id.x;
        const uint32_t outIndex = idx + (brick_size.x * brick_size.y * brick_size.z * brick_id);

        d_outCoords[outIndex].x = (brick_offset.x + d_coords[idx].x - 0.5) * lod_scale / vol_dims.x;
        d_outCoords[outIndex].y = (brick_offset.y + d_coords[idx].y - 0.5) * lod_scale / vol_dims.y;
        d_outCoords[outIndex].z = (brick_offset.z + d_coords[idx].z - 0.5) * lod_scale / vol_dims.z;

        // printf("lod: %u\n", lod_scales[brick_id]);
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void cache_use(LruContent *lru, size_t size, uint32_t *count)
    {
        uint32_t threadX = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadX >= size) return;

        if (lru[threadX].volumePosition.x != -1.f)
            atomicAdd(count, 1);
    }

    //---------------------------------------------------------------------------------------------------
    //__device__ void fill_stack(LruContent *stack, size_t size, LruContent **lrus, uint32_t *lruSizes, size_t *indexLRU)
    //{
    //    for (size_t i = 0; i < size; ++i)
    //    {
    //        stack[i] = lrus[i][(lruSizes[i] - 1) - indexLRU[i]];
    //        ++indexLRU[i];
    //    }
    //}

    //---------------------------------------------------------------------------------------------------
    //__device__ void downward(const LruContent &brick,
    //    LruContent *stack,
    //    const K_MultiResolutionPageDirectory<uint4> &mrpd,
    //    K_CacheDevice<uint4> *tableCaches,
    //    uint32_t nbTableCaches,
    //    LruContent **lrus,
    //    uint32_t *lruSizes,
    //    size_t *indexLRU)
    //{
    //    uint3 cachePosition = brick.cachePosition;
    //    const float3 &position = brick.volumePosition;
    //    uint32_t level = brick.level;

    //    stack[nbTableCaches + 1].cachePosition = cachePosition;
    //    stack[nbTableCaches + 1].level = level;
    //    stack[nbTableCaches + 1].volumePosition = position;

    //    uint4 entry = mrpd.get(level, position); //looking for the entry of the new brick

    //    int32_t i = 0;
    //    while (true)     // going down into the hierarchy to notice the first unmapped entry
    //    {
    //        if (entry.w == 1)
    //        {
    //            stack[i].cachePosition = make_uint3(entry.x, entry.y, entry.z);
    //            stack[i].level = level;
    //            stack[i].volumePosition = make_float3(-2.f, -2.f, -2.f);
    //            entry = tableCaches[i].get(entry, level, position);
    //            i++;
    //        }
    //        else
    //        {
    //            // if at least one PT cache : FILL THE STACK with the last element of each cache LRU
    //            if (nbTableCaches != 0)
    //            {
    //                fill_stack(
    //                    &stack[i == 0 ? 1 : i],
    //                    i == 0 ? nbTableCaches : nbTableCaches - i + 1,
    //                    &lrus[i == 0 ? 0 : i - 1],
    //                    lruSizes,
    //                    indexLRU);
    //            }

    //            break;
    //        }
    //    }
    //}

    //---------------------------------------------------------------------------------------------------
    //__device__ void upward(const LruContent &brick,
    //    LruContent *stack,
    //    const K_MultiResolutionPageDirectory<uint4> &mrpd,
    //    K_CacheDevice<uint4> *tableCaches,
    //    uint32_t nbTableCaches,
    //    LruContent **lrus,
    //    uint32_t *lruSizes,
    //    size_t *indexLRU,
    //    bool empty /*= false*/)
    //{
    //    const float3 &position = brick.volumePosition;
    //    uint32_t level = brick.level;

    //    int32_t i = nbTableCaches; //go to the last cache. Or 0 if only the MRPD
    //    uint4 entry;

    //    while (true)          // going up to update the hierarchy
    //    {
    //        if (stack[i].volumePosition.x < -1.f || i < 0) break; // end of loop

    //        if (i == 0)//==> mrpd
    //        {
    //            entry = mrpd.get(level, position);
    //            if (entry.w >= 1) break; //end of update - already mapped or empty
    //            const uint3 &cacheEntry = stack[i + 1].cachePosition;
    //            mrpd.set(level, position, make_uint4(
    //                cacheEntry.x,
    //                cacheEntry.y,
    //                cacheEntry.z,
    //                empty ? 2 : 1));
    //        }
    //        else
    //        {
    //            //check if we need to add a new entry or complete an existing one
    //            if (stack[i - 1].volumePosition.x < -1.f) //complete an existing one
    //            {
    //                const uint3 &cacheEntry = stack[i - 1].cachePosition;
    //                //flag the new entry
    //                tableCaches[i - 1].insert(
    //                    make_uint4(cacheEntry.x, cacheEntry.y, cacheEntry.z, 0),
    //                    level,
    //                    position,
    //                    make_uint4(stack[i + 1].cachePosition.x, stack[i + 1].cachePosition.y, stack[i + 1].cachePosition.z, 1));
    //                break;
    //            }

    //            // Entry in wich we will insert the PT of the new brick
    //            const uint3 &cacheEntry = stack[i].cachePosition;

    //            // reset entry in the current cache
    //            tableCaches[i - 1].reset_entries(cacheEntry);

    //            if (stack[i].volumePosition.x > -1.f)
    //            {
    //                //unflag from the previous cache. We must go back from the top level (the MRPD)
    //                uint4 tmpEntry = mrpd.get(stack[i].level, stack[i].volumePosition);
    //                if (i == 1)
    //                    mrpd.set(stack[i].level, stack[i].volumePosition, make_uint4(0, 0, 0, 0));
    //                else
    //                {
    //                    for (uint32_t j = 0; j < i - 1; ++j)
    //                        tmpEntry = tableCaches[j].get(tmpEntry, stack[i].level, stack[i].volumePosition);

    //                    tableCaches[i - 2].insert(
    //                        tmpEntry,
    //                        stack[i].level,
    //                        stack[i].volumePosition,
    //                        make_uint4(0, 0, 0, 0));
    //                }
    //            }

    //            //flag the new entry
    //            tableCaches[i - 1].insert(
    //                make_uint4(cacheEntry.x, cacheEntry.y, cacheEntry.z, 0),
    //                level,
    //                position,
    //                make_uint4(stack[i + 1].cachePosition.x, stack[i + 1].cachePosition.y, stack[i + 1].cachePosition.z, 1));

    //            // LRU update
    //            lrus[i - 1][(lruSizes[i - 1] - 1) - indexLRU[i - 1] + 1].volumePosition = position;
    //            lrus[i - 1][(lruSizes[i - 1] - 1) - indexLRU[i - 1] + 1].level = level;
    //        }
    //        i--;
    //    }
    //}
} // namespace gpucache
} // namespace tdns