#pragma once

#include <memory>

#include <thrust/device_vector.h>

#include <GcCore/libData/BrickKey.hpp>
#include <GcCore/libData/MetaData.hpp>
#include <GcCore/libData/Brick.hpp>
#include <GcCore/libData/BricksManager.hpp>

#include <GcCore/cuda/libCommon/DynamicArray3dHost.hpp>


namespace tdns
{
namespace preprocessor
{
    template<typename T>
    class BrickProcessor
    {
    public:

        BrickProcessor(tdns::data::MetaData &metaData);

        template<typename P>
        void process_empty(void *otherData);

        void process_histo();

    private:

        tdns::data::MetaData *_metaData;
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T, typename P>
    __global__ void kernel_process(T *bricks, bool *flags, const uint32_t nbBricks, uint32_t brickSize, void *otherData)
    {
        for (uint32_t n = blockIdx.x; n < nbBricks; n += gridDim.x)
        {
            for (uint32_t threadX = threadIdx.x; threadX < brickSize; threadX += blockDim.x)
            for (uint32_t threadY = threadIdx.y; threadY < brickSize; threadY += blockDim.y)
            for (uint32_t threadZ = threadIdx.z; threadZ < brickSize; threadZ += blockDim.z)
            {
                T voxel = bricks[(((((n * brickSize) + threadZ) * brickSize) + threadY) * brickSize) + threadX];
                if (!P::predicate(voxel, otherData))
                    flags[n] = false;
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void kernel_process_histo(uchar1 *bricks, uint64_t *histo, const uint32_t nbBricks, uint32_t brickSize)
    {
        for (uint32_t n = blockIdx.x; n < nbBricks; n += gridDim.x)
        {
            for (uint32_t threadX = threadIdx.x; threadX < brickSize; threadX += blockDim.x)
            for (uint32_t threadY = threadIdx.y; threadY < brickSize; threadY += blockDim.y)
            for (uint32_t threadZ = threadIdx.z; threadZ < brickSize; threadZ += blockDim.z)
            {
                uchar1 voxel = bricks[(((((n * brickSize) + threadZ) * brickSize) + threadY) * brickSize) + threadX];
                atomicAdd(reinterpret_cast<unsigned long long int*>(&histo[voxel.x]), static_cast<unsigned long long int>(1));
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void kernel_process_histo(ushort1 *bricks, uint64_t *histo, const uint32_t nbBricks, uint32_t brickSize)
    {
        for (uint32_t n = blockIdx.x; n < nbBricks; n += gridDim.x)
        {
            for (uint32_t threadX = threadIdx.x; threadX < brickSize; threadX += blockDim.x)
            for (uint32_t threadY = threadIdx.y; threadY < brickSize; threadY += blockDim.y)
            for (uint32_t threadZ = threadIdx.z; threadZ < brickSize; threadZ += blockDim.z)
            {
                ushort1 voxel = bricks[(((((n * brickSize) + threadZ) * brickSize) + threadY) * brickSize) + threadX];
                atomicAdd(reinterpret_cast<unsigned long long int*>(&histo[voxel.x]), static_cast<unsigned long long int>(1));
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    BrickProcessor<T>::BrickProcessor(tdns::data::MetaData &metaData)
    {
        _metaData = &metaData;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    template<typename P>
    void BrickProcessor<T>::process_empty(void *otherData)
    {
        const uint32_t nbBricks = 1;
        std::vector<tdns::data::Bkey> &emptyBricks = _metaData->get_empty_bricks();
        //to get meta data like: encoded and size
        //tdns::data::Brick *brick = tdns::data::BricksManager::get_instance()->get_brick(0, { 0, 0, 0 });
        tdns::data::Brick *brick = nullptr; //<< NEED TO FIX THIS
        // tdns::data::Brick *brick = nullptr;
        if (!brick) return;
        tdns::math::Vector3ui brick_edge_size = brick->get_edge_size();

        tdns::common::DynamicArray3dHost<T, tdns::common::DynamicArrayOptions::Options::Mapped> buffer
            (tdns::math::Vector3ui(nbBricks * brick_edge_size[0], brick_edge_size[1], brick_edge_size[2]));
        T *mappedPtr = nullptr;
        buffer.get_mapped_device_pointer(&mappedPtr);

        thrust::device_vector<bool> emptyFlag(nbBricks, true);

        const dim3 nbThreadPerBlock(8, 8, 8);
        // const dim3 nbBlocks((nbBricks * brick_edge_size) / nbThreadPerBlock.x,
        // brick_edge_size / nbThreadPerBlock.y,
        // brick_edge_size / nbThreadPerBlock.z);

        std::vector<tdns::math::Vector3ui> &nbBricksPerLevel = _metaData->get_nb_bricks();
        std::vector<tdns::math::Vector3ui> &nbBigBricksPerLevel = _metaData->get_nb_big_bricks();

        for (uint32_t l = 0; l < nbBricksPerLevel.size(); ++l)
        {
            tdns::math::Vector3ui &totalBricks = nbBigBricksPerLevel[l];

            for (uint32_t z = 0; z < totalBricks[0]; ++z)
            {
                for (uint32_t y = 0; y < totalBricks[1]; ++y)
                {
                    for (uint32_t x = 0; x < totalBricks[2]; ++x)
                    {
                        thrust::fill(emptyFlag.begin(), emptyFlag.end(), true);
                        tdns::math::Vector3ui position(x, y, z);
                        //brick = tdns::data::BricksManager::get_instance()->get_brick(l, position);
                        brick = nullptr;
                        if (!brick) continue;
                        std::memcpy(&buffer[0], brick->get_data().data(), brick->get_data().size());

                        // static_cast<uint32_t>(brick_edge_size[0]) for the moment
                        // but it should pass all three dimensions.
                        kernel_process<T, P><<<nbBricks, nbThreadPerBlock>>>(mappedPtr, emptyFlag.data().get(), nbBricks, static_cast<uint32_t>(brick_edge_size[0]), otherData);

                        if(emptyFlag[0]) //empty
                            emptyBricks.push_back(tdns::data::get_key(l, {x, y, z}));
                    }//for x
                }//for y
            }//for z
        }//for l
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    void BrickProcessor<T>::process_histo()
    {
        const uint32_t nbBricks = 1;
        //tdns::data::Brick *brick = tdns::data::BricksManager::get_instance()->get_brick(0, { 0, 0, 0 });
        tdns::data::Brick *brick = nullptr;
        if (!brick) return;
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        uint32_t brickSize, covering;
        conf.get_field("BrickSize", brickSize);
        conf.get_field("VoxelCovering", covering);
        uint32_t usedBrick = brickSize - covering;
        size_t brickLength = sizeof(T) * usedBrick * usedBrick * usedBrick;

        tdns::common::DynamicArray3dHost<T, tdns::common::DynamicArrayOptions::Options::Mapped> buffer
            (tdns::math::Vector3ui(nbBricks * usedBrick, usedBrick, usedBrick));
        T *mappedPtr = nullptr;
        buffer.get_mapped_device_pointer(&mappedPtr);

        std::vector<float> &histoFloat = _metaData->get_histo();
        size_t histoSize = histoFloat.size();
        std::vector<uint64_t> histo(histoSize, 0);
        uint64_t *d_histo;
        CUDA_SAFE_CALL(cudaMalloc(&d_histo, histoSize * sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMemset(d_histo, 0, histoSize * sizeof(uint64_t)));

        const dim3 nbThreadPerBlock(8, 8, 8);

        std::vector<tdns::math::Vector3ui> &nbBricksPerLevel = _metaData->get_nb_bricks();
        tdns::math::Vector3ui &totalBricks = nbBricksPerLevel[0];

        for (uint32_t z = 0; z < totalBricks[0]; ++z)
        {
            for (uint32_t y = 0; y < totalBricks[1]; ++y)
            {
                for (uint32_t x = 0; x < totalBricks[2]; ++x)
                {
                    tdns::math::Vector3ui position(x, y, z);
                    //brick = tdns::data::BricksManager::get_instance()->get_brick(0, position);
                    brick = nullptr;
                    if (!brick) continue;
                    std::memcpy(&buffer[0], brick->get_data().data(), brickLength);

                    kernel_process_histo<<<nbBricks, nbThreadPerBlock>>>(mappedPtr, d_histo, nbBricks, usedBrick);

                }//for x
            }//for y
        }//for z
        
        CUDA_SAFE_CALL(cudaMemcpy(histo.data(), d_histo, histoSize * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        tdns::math::Vector3ui &initialSize = _metaData->get_initial_size(0);
        size_t nbPixels = initialSize[0] * initialSize[1] * initialSize[2];

        for (uint32_t i = 0; i < histoFloat.size(); ++i)
            histoFloat[i] = (std::log(histo[i]) - std::log(1)) / (std::log(nbPixels) - std::log(1));
    }
} // namespace preprocessor
} // namespace tdns