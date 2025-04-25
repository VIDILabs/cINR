#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <climits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <thrust/shuffle.h>
// #include <thrust/random.h>
#include <thrust/sort.h>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/KernelObject.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dDevice.hpp>
#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libData/VolumeConfiguration.hpp>
#include <GcCore/libData/CacheConfiguration.hpp>

#include <GcCore/cuda/libGPUCache/K_CacheManager.hpp>
#include <GcCore/cuda/libGPUCache/K_CacheDevice.hpp>
#include <GcCore/cuda/libGPUCache/K_MultiResolutionPageDirectory.hpp>
#include <GcCore/cuda/libGPUCache/MultiResolutionPageDirectory.hpp>
#include <GcCore/cuda/libGPUCache/RequestHandler.hpp>
#include <GcCore/cuda/libGPUCache/CacheManagerKernels.hpp>

namespace tdns
{
namespace gpucache
{
    /**
     * @brief
     */
    template<typename T>
    class CacheManager : public 
        tdns::common::KernelObject<K_CacheManager<T>>,
        tdns::common::Noncopyable
    {
    public:

        /**
         * @brief Constructor.
         * @param[in]   cacheSize       3D size of the caches for each cache in the Page Table hierarchy.
         * @param[in]   blockSize       3D size of the virtualized element/brick for each in the Page Table hierarchy.
         * @param[in]   volumeSizes     3D size of the volume for each level of resolution.
         */
        CacheManager(const tdns::data::VolumeConfiguration &volumeConfiguration, const tdns::data::CacheConfiguration &cacheConfiguration, int32_t gpuID = 0);

        /**
         * @brief Destructor.
         */
        virtual ~CacheManager();

        /**
         * 
         */
        virtual K_CacheManager<T> to_kernel_object() override;

        /**
         * @brief 
         */
        bool update();

        /**
        * @brief
        */
        void completude(std::vector<float> &vec);

    protected:

        /**
         * @brief
         */
        void lru_update();

        /**
         * @brief
         * 
         * @param mask [description]
         */
        template <typename U>
        void create_mask(thrust::device_vector<bool> &mask, CacheDevice<U> &cache);

        /**
         * @brief
         */
        void create_request_list(thrust::device_vector<uint4> &requestedBricks);

        /**
        *
        */
        bool prepare_bricks();
        
        void end_of_load_callback();

        void load_new_bricks();

    public:

        /**
        * Member data
        */
        uint32_t                                                        _timestamp;             ///<
        std::unique_ptr<MultiResolutionPageDirectory<uint4>>            _mrpd;                  ///<
        std::unique_ptr<CacheDevice<T>>                                 _dataCache;             ///<
        std::vector<std::unique_ptr<CacheDevice<uint4>>>                _tableCaches;           ///<
        std::unique_ptr<tdns::common::DynamicArray3dDevice<uint32_t>>   _requestBuffer;         ///<
        std::unique_ptr<tdns::common::DynamicArray3dDevice<uint8_t>>    _dataCacheMask;         ///<
        std::unique_ptr<RequestHandler<T>>                              _requestHandler;        ///<
        size_t                                                          _nbMaxRequests;         ///<
        tdns::math::Vector3f                                            _oneOverBrickSize;      ///<
        std::unique_ptr<tdns::common::DynamicArray3dDevice<float3>>     _initialOverRealSize;   ///<

        tdns::math::Vector3ui                                           _brickSize;             ///<
        tdns::math::Vector3ui                                           _covering;              ///<

        // vnrNetwork                                                      _net;                   ///< Injected model network
        // vnr::vec3f                                                      _volDims;               ///< Volume dims for coordinate normalization
        // vnr::vec3f*                                                     _d_coords;              ///< Coordinates to be offset for VNR on Device
        // vnr::vec3f*                                                     _d_outCoords;           ///< Coordinates for inference
        // float*                                                          _d_values;              ///< Output values from network

    private:
        
        K_CacheDevice<uint4>                                            *_k_tableCache;     ///<
        std::mutex                                                      _mutexDone;         ///<
        bool                                                            _requestDone;       ///<

        std::chrono::high_resolution_clock::time_point _startLoading, _endLoading;
    };

    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline CacheManager<T>::CacheManager(const tdns::data::VolumeConfiguration &volumeConfiguration, const tdns::data::CacheConfiguration &cacheConfiguration, int32_t gpuID /*= 0*/)
        : _brickSize(volumeConfiguration.BrickSize),
        _covering(volumeConfiguration.Covering)
    {
        const std::vector<tdns::math::Vector3ui> &cacheSize = cacheConfiguration.CacheSize;
        const std::vector<tdns::math::Vector3ui> &blockSize = cacheConfiguration.BlockSize;
        const std::vector<tdns::math::Vector3ui> &volumeSizes = volumeConfiguration.RealVolumesSizes;

        LOGINFO(40, tdns::common::log_details::Verbosity::INSANE, "Creating CacheManager.");
#if TDNS_OS != TDNS_OS_WINDOWS //erreur de compile sur windows... il faut regarder ca...
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " |- cacheSize = [" << cacheSize.size() << "]");
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " |- blockSize = [" << blockSize.size() << "]");
        LOGDEBUG(40, tdns::common::log_details::Verbosity::INSANE, " `- volumeSizes = [" << volumeSizes.size() << "]");
#endif

        _tableCaches.resize(cacheSize.size() - 1);
        _timestamp = 0;

        uint32_t nbLevelOfResolution = volumeSizes.size();
        uint32_t nbLevelPTHierarchy = cacheSize.size() + 1;

        std::vector<std::vector<uint3>> realNumberOfEntries(nbLevelPTHierarchy, std::vector<uint3>(nbLevelOfResolution));
        std::vector<std::vector<uint3>> virtualizedNumberOfEntries(nbLevelPTHierarchy, std::vector<uint3>(nbLevelOfResolution));
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        /* Compute real number of entries */
        // INIT -> The number of entries in the first cache is computed from the size of the volume.
        // The number of entries of other caches are computed from the previous cache in the PT hierarchy (*)        
        std::memcpy(realNumberOfEntries[0].data(), volumeSizes.data(), volumeSizes.size() * sizeof(tdns::math::Vector3ui));
        for (uint32_t j = 0; j < nbLevelOfResolution; ++j)  // For each level of resolution -> 
        {
            // (*)
            realNumberOfEntries[1][j].x = static_cast<uint32_t>(std::ceil(static_cast<float>(realNumberOfEntries[0][j].x) / (blockSize[0][0] - 2.0 * volumeConfiguration.Covering[0])));
            realNumberOfEntries[1][j].y = static_cast<uint32_t>(std::ceil(static_cast<float>(realNumberOfEntries[0][j].y) / (blockSize[0][1] - 2.0 * volumeConfiguration.Covering[1])));
            realNumberOfEntries[1][j].z = static_cast<uint32_t>(std::ceil(static_cast<float>(realNumberOfEntries[0][j].z) / (blockSize[0][2] - 2.0 * volumeConfiguration.Covering[2])));
        }
        for (uint32_t i = 2; i < nbLevelPTHierarchy; ++i)   // For each cache
        for (uint32_t j = 0; j < nbLevelOfResolution; ++j)  // For each level of resolution -> 
        {
            // (*)
            realNumberOfEntries[i][j].x = static_cast<uint32_t>(std::ceil(static_cast<float>(realNumberOfEntries[i - 1][j].x) / blockSize[i - 1][0]));
            realNumberOfEntries[i][j].y = static_cast<uint32_t>(std::ceil(static_cast<float>(realNumberOfEntries[i - 1][j].y) / blockSize[i - 1][1]));
            realNumberOfEntries[i][j].z = static_cast<uint32_t>(std::ceil(static_cast<float>(realNumberOfEntries[i - 1][j].z) / blockSize[i - 1][2]));
        }
        /////////////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////////////        
        /* Compute virtual number of entries */
        // INIT -> The virtual number of entries of the last (.back) cache is computed from the MRPD dimension
        // The virtual number of entries of the other caches are computed from the previous cache in the PT hierarchy (*)
        virtualizedNumberOfEntries.back() = realNumberOfEntries.back();

        for (uint32_t i = 1; i < nbLevelPTHierarchy - 1; ++i)   // For each cache
        for (uint32_t j = 0; j < nbLevelOfResolution; ++j)  // For each level of resolution ->
        {
            // (*)
            uint32_t current = nbLevelPTHierarchy - 1 - i;

            virtualizedNumberOfEntries[current][j].x = static_cast<uint32_t>(
                std::ceil(static_cast<float>(virtualizedNumberOfEntries[current + 1][j].x) * blockSize[current][0]));
            virtualizedNumberOfEntries[current][j].y = static_cast<uint32_t>(
                std::ceil(static_cast<float>(virtualizedNumberOfEntries[current + 1][j].y) * blockSize[current][1]));
            virtualizedNumberOfEntries[current][j].z = static_cast<uint32_t>(
                std::ceil(static_cast<float>(virtualizedNumberOfEntries[current + 1][j].z) * blockSize[current][2]));
        }
        for (uint32_t j = 0; j < nbLevelOfResolution; ++j)  // For each level of resolution ->
        {
            virtualizedNumberOfEntries[0][j].x = static_cast<uint32_t>(
                std::ceil(static_cast<float>(virtualizedNumberOfEntries[1][j].x) * (blockSize[0][0] - 2.0 * volumeConfiguration.Covering[0])));
            virtualizedNumberOfEntries[0][j].y = static_cast<uint32_t>(
                std::ceil(static_cast<float>(virtualizedNumberOfEntries[1][j].y) * (blockSize[0][1] - 2.0 * volumeConfiguration.Covering[1])));
            virtualizedNumberOfEntries[0][j].z = static_cast<uint32_t>(
                std::ceil(static_cast<float>(virtualizedNumberOfEntries[1][j].z) * (blockSize[0][2] - 2.0 * volumeConfiguration.Covering[2])));
        }
        /////////////////////////////////////////////////////////////////////////////////////////////////////        
        
        // Create the Multi-Resolution Page Directory
        _mrpd = tdns::common::create_unique_ptr<MultiResolutionPageDirectory<uint4>>
            (realNumberOfEntries.back());
        
        // Create the brick cache
        _dataCache = tdns::common::create_unique_ptr<CacheDevice<T>>
            (cacheSize.front(),
            blockSize.front(),
            virtualizedNumberOfEntries.front(),
            realNumberOfEntries[0],
            &_timestamp,
            cacheConfiguration.DataCacheFlags);

         // Create the caches of PT
        for (uint32_t i = 0; i < nbLevelPTHierarchy - 2; ++i) // nbLevelPTHierarchy - 2 : without the MRPD and the data_cache
        {
            uint32_t index = nbLevelPTHierarchy - 2 - i;

            _tableCaches[i] = tdns::common::create_unique_ptr<CacheDevice<uint4>>
                (cacheSize[index],
                blockSize[index],
                virtualizedNumberOfEntries[index],
                realNumberOfEntries[index],
                &_timestamp);
        }

        // Create the request buffer
        tdns::math::Vector3ui size(0);
        // realNumberOfEntries[1] -> because the request buffer is mapped on the last level of PT hierarchy (not the data cache)
        for (auto it = realNumberOfEntries[1].begin(); it != realNumberOfEntries[1].end(); ++it)
        {
            size[0] += it->x;
            size[1] = std::max(size[1], it->y);
            size[2] = std::max(size[2], it->z);
        }

        // As many entry as bricks in the volume (in each level of resolution)
        _requestBuffer = tdns::common::create_unique_ptr<tdns::common::DynamicArray3dDevice<uint32_t>>(size);
        cudaMemset(_requestBuffer->data(), 0, size[0]*size[1]*size[2]*sizeof(uint32_t));

        size = cacheSize.front() * blockSize.front();
        // size[0] = size[1] = size[2] = 0;
        
        // As many entry as voxel in the dataCache
        _dataCacheMask = tdns::common::create_unique_ptr<tdns::common::DynamicArray3dDevice<uint8_t>>(size, 0);

        // Create a tableCache array (pointer) for the K_CacheManager.
        CUDA_SAFE_CALL(cudaMalloc(&_k_tableCache, _tableCaches.size() * sizeof(K_CacheDevice<uint4>)));

        const tdns::math::Vector3ui &brickSize = blockSize.front();

        // Number of brick requests is limited to 50 !
        _nbMaxRequests = 40;
        _requestHandler = tdns::common::create_unique_ptr<RequestHandler<T>>(volumeConfiguration, _nbMaxRequests, gpuID);
        _requestDone = false;
        _requestHandler->start();

        for (uint32_t i = 0; i < 3; ++i)
            _oneOverBrickSize[i] = 1.f / static_cast<float>(brickSize[i] - 2 * volumeConfiguration.Covering[i]);

        //send the real and initial volume size to consider the black induce by the bricking step.
        const std::vector<tdns::math::Vector3ui> &initialVolumeSizes = volumeConfiguration.InitialVolumeSizes;
        size = tdns::math::Vector3ui(initialVolumeSizes.size(), 1, 1);
        _initialOverRealSize = tdns::common::create_unique_ptr<tdns::common::DynamicArray3dDevice<float3>>(size);
        std::vector<float3> initialOverReal(initialVolumeSizes.size());
        for (size_t i = 0; i < initialVolumeSizes.size(); ++i)
        {
            float x = static_cast<float>(initialVolumeSizes[i][0]) / static_cast<float>(volumeSizes[i][0]);
            float y = static_cast<float>(initialVolumeSizes[i][1]) / static_cast<float>(volumeSizes[i][1]);
            float z = static_cast<float>(initialVolumeSizes[i][2]) / static_cast<float>(volumeSizes[i][2]);

            initialOverReal[i] = make_float3(x, y, z);
        }

        *_initialOverRealSize = initialOverReal;

        // vnr::vec3f *h_coords;
        // h_coords = new vnr::vec3f[brickSize[0]*brickSize[1]*brickSize[2]];

        // int n = 0;
        // for (uint i=0; i<brickSize[2]; ++i) {
        //     for (uint j=0; j<brickSize[1]; ++j) {
        //         for (uint k=0; k<brickSize[0]; ++k) {
        //             h_coords[n] = vnr::vec3f(k,j,i);
        //             n++;
        //         }
        //     }
        // }
        // const uint N = n * sizeof(vnr::vec3f);

        // cudaMalloc(&_d_coords, N);
        // cudaMalloc(&_d_outCoords, N * _nbMaxRequests);
        // cudaMemcpy(_d_coords, h_coords, N, cudaMemcpyHostToDevice);

        // cudaMalloc(&_d_values, n * sizeof(T) * _nbMaxRequests);

        // // cudaMalloc(&_d_outCoords, n * sizeof(vnr::vec3f) * 50); // hardcoded max num requests for both for now
        // // cudaMalloc(&_d_values, n * sizeof(float) * 50);

        // delete[] h_coords;

        // vnrJson root = vnrCreateJsonBinary(volumeConfiguration.VolumeDirectory + "/params.json");

        // _net = vnrCreateNetwork(root);

        // if (root.contains("volume")) {
        //     _volDims = vnr::vec3f(root["volume"]["dims"]["x"].get<int>(),
        //                           root["volume"]["dims"]["y"].get<int>(),
        //                           root["volume"]["dims"]["z"].get<int>());
            
        //     // Manual dims saving for OUT-OF-CORE trained params

        //     //     root["volume"] = {
        //     //         { "dims", {
        //     //         { "x", 2048 },
        //     //         { "y", 2048 },
        //     //         { "z", 1920 }
        //     //         }}
        //     // //     };

        //     //     const auto broot = vnrJson::to_bson(root);
        //     //     std::ofstream ofs("new_params.json", std::ios::binary | std::ios::out);
        //     //     ofs.write((char*)broot.data(), broot.size());
        //     //     ofs.close();

        //     // End of Manual saving

        //     std::cout << "Volume Dims: "<< _volDims.x << ' ' << _volDims.y << ' ' << _volDims.z << std::endl;
        // }

    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline CacheManager<T>::~CacheManager()
    {
        _requestHandler->stop();
        CUDA_SAFE_CALL_NOEXCEPT(cudaFree(_k_tableCache));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline K_CacheManager<T> CacheManager<T>::to_kernel_object()
    {
        size_t nbtableCaches = _tableCaches.size();

        for (size_t i = 0; i < nbtableCaches; ++i)
        {
            K_CacheDevice<uint4> tmp = _tableCaches[i]->to_kernel_object();

            CUDA_SAFE_CALL(cudaMemcpy(
                &_k_tableCache[i],
                &tmp,
                sizeof(K_CacheDevice<uint4>),
                cudaMemcpyHostToDevice));
        }

        return K_CacheManager<T>(*_mrpd, *_dataCache, _k_tableCache, nbtableCaches, *_requestBuffer, *_dataCacheMask, *_initialOverRealSize, _timestamp, _brickSize[0], _covering[0]);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline bool CacheManager<T>::update()
    {
        lru_update();

        //not ready to load new bricks and we did not load the last request.
        if (_requestHandler->is_working())
        {
            _timestamp += 1000;
            return false;
        }

        bool check;

        //bricks to load to the GPU ?
        if (_requestDone)
        {
            load_new_bricks();
            std::lock_guard<std::mutex> guard(_mutexDone);
            _requestDone = false;

            check = false;

            _endLoading = std::chrono::high_resolution_clock::now();
            long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(_endLoading - _startLoading).count();

            //std::cout << "Time to load [" << _requestHandler->get_asked_bricks().size() << "] bricks in ["<< duration <<"] ms." << std::endl;
        }
        else
        {
            _startLoading = std::chrono::high_resolution_clock::now();
            check = prepare_bricks();
        }
        _timestamp += 1000;

        return check;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void CacheManager<T>::completude(std::vector<float> &vec)
    {
        thrust::device_vector<LruContent>& lru = _dataCache->get_lru();
        size_t lruSize = lru.size();
        // Prepare the kernel launch
        const dim3 nbThreadsPerBlock(std::min(128U, static_cast<uint32_t>(lruSize)), 1, 1);
        uint32_t numBlock = static_cast<uint32_t>((lruSize % nbThreadsPerBlock.x != 0) ?
            (lruSize / nbThreadsPerBlock.x + 1) :
            (lruSize / nbThreadsPerBlock.x));
        const dim3 nbBlocks(numBlock, 1, 1);
            
        uint32_t count;
        uint32_t *d_count;
        CUDA_SAFE_CALL(cudaMalloc(&d_count, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(uint32_t)));
        
        cache_use<<<nbBlocks, nbThreadsPerBlock>>>(lru.data().get(), lruSize, d_count);
#if TDNS_MODE == TDNS_MODE_DEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
        CUDA_CHECK_KERNEL_ERROR();

        CUDA_SAFE_CALL(cudaMemcpy(&count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        vec.push_back(static_cast<float>(count) / static_cast<float>(lruSize));
        CUDA_SAFE_CALL(cudaFree(d_count));
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void CacheManager<T>::lru_update()
    {   
        // Create the usage mask
        thrust::device_vector<bool> mask;

        // For the data cache
        create_mask(mask, *_dataCache);

        // Update the LRU data cache with the mask
        _dataCache->update(mask, true);

        // For each PT caches
        for (auto it = _tableCaches.begin(); it != _tableCaches.end(); ++it)
        {
            create_mask(mask, **it);
            // Update the LRU cache with the mask
            (*it)->update(mask, true);
        }
    }

    //---------------------------------------------------------------------------------------------------

    template<typename T>
    inline void CacheManager<T>::create_request_list(thrust::device_vector<uint4> &requestedBricks)
    {
        static thrust::device_vector<size_t> result(_requestBuffer->size());
        static thrust::device_vector<uint32_t> temp_requests(_requestBuffer->size());

        // stream compaction to keep the flaged elements (with curent timestamp)
        size_t nbElements = thrust::copy_if
        (
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(_requestBuffer->size() - 1),
            _requestBuffer->begin(),
            result.begin(),
            predicate::is_greater_equal<uint32_t>(_timestamp)
        ) - result.begin();

        // if (_timestamp % 10 == 0){
        //     std::cout << nbElements << ' ' << _timestamp << std::endl << std::endl;
        // }

        // static thrust::default_random_engine rng;
        // thrust::shuffle(result.begin(), result.begin()+nbElements, rng);

        // thrust::sort(result.begin(), result.begin() + nbElements, thrust::greater<size_t>());
        thrust::device_vector<size_t> temp_result = result;
        thrust::copy(_requestBuffer->begin(), _requestBuffer->end(), temp_requests.begin());

        // auto sort_permute = thrust::make_permutation_iterator(_requestBuffer->begin(), temp.begin());

        thrust::sort_by_key(
            thrust::make_permutation_iterator(temp_requests.begin(), temp_result.begin()),
            thrust::make_permutation_iterator(temp_requests.begin(), temp_result.begin()+nbElements),
            result.begin(),
            thrust::greater<size_t>()
        );

        // thrust::host_vector<size_t> h_result = result;
        // thrust::host_vector<uint32_t> h_buffer(_requestBuffer->size());
        // thrust::copy(_requestBuffer->begin(), _requestBuffer->end(), h_buffer.begin());

        // cudaDeviceSynchronize();

        // for (int i=0; i<50; i++) {
        //     std::cout << i << ": Brick_id: " << h_result[i] << " | Rank: " << h_buffer[h_result[i]] << std::endl;
        // }
        
        // std::cout << "TimeStamp: " << _timestamp << std::endl;

        // for (int i=0; i < h_result.size(); i++) {
        //     if (h_result[i] >= _requestBuffer->size())
        //         printf("%i: %lu\n", i, h_result[i]);
        // }




        // To mesure the time to handle a list of bricks requests
        static bool done = false;
        static std::chrono::high_resolution_clock::time_point start, end;
        if(nbElements == 0)
        {
            if (done)
            {
                end = std::chrono::high_resolution_clock::now();
                uint64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                // std::cout << duration << " us" << std::endl;
                done = false;
            }
            return;
        }
        if (!done) start = std::chrono::high_resolution_clock::now();
        done = true;

        // limit the number of request brick !
        size_t maxElem = _nbMaxRequests;
        nbElements = std::min(nbElements, maxElem);

        // Request list with 3D indexes and level of resolution
        requestedBricks.resize(nbElements);


        // Prepare the kernel launch
        const dim3 nbThreadsPerBlock(std::min(128U, static_cast<uint32_t>(nbElements)), 1, 1);
        uint32_t numBlock = static_cast<uint32_t>((nbElements % nbThreadsPerBlock.x != 0) ?
            (nbElements / nbThreadsPerBlock.x + 1) :
            (nbElements / nbThreadsPerBlock.x));
        // NOTE : try in 2 dimensions instead of 1 (with a max limit (65536U ?) on the first dimension)
        const dim3 nbBlocks(numBlock, 1, 1);

        tdns::common::K_DynamicArray3dDevice<uint3> realNumberOfEntries;

        if (_tableCaches.size() == 0)   // -> MRPD only
            realNumberOfEntries = _mrpd->get_level_dimensions().to_kernel_object();
        else
            realNumberOfEntries = _tableCaches.back()->get_real_number_of_entries().to_kernel_object();

        coord_1D_to_3D<<<nbBlocks, nbThreadsPerBlock>>>(
                result.data().get(),
                nbElements,
                requestedBricks.data().get(),
                realNumberOfEntries);
#if TDNS_MODE == TDNS_MODE_DEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
        CUDA_CHECK_KERNEL_ERROR();

        return;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline bool CacheManager<T>::prepare_bricks()
    {
        // =========================== REQUEST ===========================
        thrust::device_vector<uint4> requestedBricks;
        
        create_request_list(requestedBricks);

        const uint32_t nbBricks = static_cast<uint32_t>(requestedBricks.size());
        if (nbBricks == 0) return true;

        _requestHandler->notify_request(requestedBricks, [&] () { this->end_of_load_callback(); });/*std::bind(&CacheManager::end_of_load_callback, this));*/

        return false;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void CacheManager<T>::end_of_load_callback()
    {
        std::lock_guard<std::mutex> guard(_mutexDone);
        _requestDone = true;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void CacheManager<T>::load_new_bricks()
    {
        static thrust::device_vector<uint4> &requestedBricks = _requestHandler->get_asked_bricks();
        uint32_t nbBricks = static_cast<uint32_t>(requestedBricks.size());
        if (nbBricks == 0) return;
        uint32_t nbNonEmptyBricks = static_cast<uint32_t>(_requestHandler->get_nb_non_empty_bricks());

        // Positions of the bricks that will be remove by the new requested bricks. (The ones at the end of the LRU)
        static thrust::device_vector<LruContent> &dataCacheLRU = _dataCache->get_lru();
        LruContent *endDataCacheLRUpositions = dataCacheLRU.data().get() + (dataCacheLRU.size() - nbNonEmptyBricks);
        
        // cuda kernel configuration
        dim3 nbThreadsPerBlock;
        uint32_t numBlock;
        dim3 nbBlocks;

        if (nbNonEmptyBricks != 0) //non empty bricks to handle
        {
            // ======================= UPDATE PT HIERARCHY ===================
            // Prepare the kernel launch
            nbThreadsPerBlock = dim3(std::min(128U, nbNonEmptyBricks), 1, 1);
            numBlock = (nbNonEmptyBricks % nbThreadsPerBlock.x != 0) ?
                (nbNonEmptyBricks / nbThreadsPerBlock.x + 1) :
                (nbNonEmptyBricks / nbThreadsPerBlock.x);
            // NOTE : try in 2 dimensions instead of 1 (with a max limit (65536U ?) on the first dimension)
            nbBlocks = dim3(numBlock, 1, 1);

            // Dereference the bricks that will be remove.
            dereference_old_bricks<<<nbBlocks, nbThreadsPerBlock>>>(this->to_kernel_object(), endDataCacheLRUpositions, nbNonEmptyBricks);
#if TDNS_MODE == TDNS_MODE_DEBUG
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
            CUDA_CHECK_KERNEL_ERROR();
        }
            // std::cout << "-------------------------\n";

        // ====================== UPDATE LRU / BRICKS POSITIONS ==================
        uint3 brickSize = *reinterpret_cast<const uint3*>(_requestHandler->get_brick_size().data());

        thrust::device_vector<LruContent> bricksPositionsDraft(nbBricks);

        // Prepare the kernel launch
        nbThreadsPerBlock = dim3(std::min(128U, nbBricks), 1, 1);
        numBlock = (nbBricks % nbThreadsPerBlock.x != 0) ? (nbBricks / nbThreadsPerBlock.x + 1) : (nbBricks / nbThreadsPerBlock.x);
        nbBlocks = dim3(numBlock, 1, 1);

        // Compute the data cache LRU with the volume positions (and level) of the new bricks that will be inserted.
        compute_lru_volume_position<<<nbBlocks, nbThreadsPerBlock>>>(
            bricksPositionsDraft.data().get(),
            _dataCache->get_virtual_number_of_entries().to_kernel_object(),
            requestedBricks.data().get(),
            nbBricks,
            *reinterpret_cast<const float3*>(_oneOverBrickSize.data()),
            std::numeric_limits<float>::epsilon()
            );

        if (nbNonEmptyBricks != 0) //non empty bricks to handle
        {
            // Update the data cache LRU positions
            nbThreadsPerBlock = dim3(std::min(128U, nbNonEmptyBricks), 1, 1);
            numBlock = (nbNonEmptyBricks % nbThreadsPerBlock.x != 0) ? (nbNonEmptyBricks / nbThreadsPerBlock.x + 1) : (nbNonEmptyBricks / nbThreadsPerBlock.x);
            nbBlocks = dim3(numBlock, 1, 1);
            copy_lru_position<<<nbBlocks, nbThreadsPerBlock>>>(endDataCacheLRUpositions, bricksPositionsDraft.data().get(), nbNonEmptyBricks);

#if TDNS_MODE == TDNS_MODE_DEBUG
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
            CUDA_CHECK_KERNEL_ERROR();
            // ====================== WRITE BRICKS IN CACHE ==================
            // T *bricks = nullptr;
            static T* bricks = (T*)_requestHandler->_bricksManager->_d_values;
            // static T* bricks = (T*)_d_values;
            // _requestHandler->get_request_buffer().get_mapped_device_pointer(&bricks);

            // dim3 blockDim(16, 8, 8);
            // dim3 gridDim(((brickSize.x * _nbMaxRequests) + blockDim.x -1) / blockDim.x,
            //              ((brickSize.y * _nbMaxRequests) + blockDim.y -1) / blockDim.y,
            //              ((brickSize.z * _nbMaxRequests) + blockDim.z -1) / blockDim.z);

            // fillAllCoords<<<gridDim,blockDim>>>(brickSize, requestedBricks.data().get(), _volDims, _nbMaxRequests, _d_coords, _d_outCoords);
            // CUDA_SYNC_CHECK();

            // vnrInfer(_net, (vnrDevicePtr)(_d_outCoords), (vnrDevicePtr)(_d_values), brickSize.x * brickSize.y * brickSize.z * _nbMaxRequests);

            tdns::math::Vector3ui bigBrickSize = _requestHandler->get_big_brick_size();
            tdns::math::Vector3ui bigBrickVoxels;
            bigBrickVoxels[0] = bigBrickSize[0] * brickSize.x;
            bigBrickVoxels[1] = bigBrickSize[1] * brickSize.y;
            bigBrickVoxels[2] = bigBrickSize[2] * brickSize.z;

            thrust::device_vector<uint32_t> bigBrickIndexes = _requestHandler->get_big_brick_indexes();
            thrust::device_vector<uint4> bigBrickCoords = _requestHandler->get_big_brick_coords();

            nbThreadsPerBlock = dim3(8, 8, 8);
            nbBlocks = dim3(std::ceil(brickSize.x / static_cast<float>(nbThreadsPerBlock.x)), std::ceil(brickSize.y / static_cast<float>(nbThreadsPerBlock.y)), std::ceil(brickSize.z / static_cast<float>(nbThreadsPerBlock.z)));

            write_brick <<<nbBlocks, nbThreadsPerBlock>>> (
                _dataCache->to_kernel_object(),
                bricks, endDataCacheLRUpositions,
                requestedBricks.data().get(),
                bigBrickIndexes.data().get(),
                bigBrickCoords.data().get(),
                nbNonEmptyBricks, brickSize,
                *reinterpret_cast<const uint3*>(bigBrickSize.data()),
                *reinterpret_cast<const uint3*>(bigBrickVoxels.data()));

#if TDNS_MODE == TDNS_MODE_DEBUG
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
            CUDA_CHECK_KERNEL_ERROR();
        }
        // ====================== REFERENCE NEW BRICKS ==================
        /// @todo optimiser ! Peut etre fait qu'une fois dans le constructeur
        uint32_t nbTableCaches = static_cast<uint32_t>(_tableCaches.size());
        thrust::device_vector<LruContent*> d_lrus(nbTableCaches);
        thrust::device_vector<uint32_t> d_lruSizes(nbTableCaches);

        for (uint32_t i = 0; i < nbTableCaches; ++i)
        {
            thrust::device_vector<LruContent> &lru = _tableCaches[i]->get_lru();
            d_lrus[i] = lru.data().get();
            d_lruSizes[i] = static_cast<uint32_t>(lru.size());
        }

        // Reference the new bricks that will be added into the cache
        reference_new_bricks<<<1, 1>>>(
            this->to_kernel_object(),
            endDataCacheLRUpositions,
            d_lrus.data().get(),
            bricksPositionsDraft.data().get() + nbNonEmptyBricks,
            d_lruSizes.data().get(),
            nbBricks - nbNonEmptyBricks,
            nbNonEmptyBricks);
#if TDNS_MODE == TDNS_MODE_DEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
        CUDA_CHECK_KERNEL_ERROR();
    }

    //---------------------------------------------------------------------------------------------------
    template <typename T>
    template <typename U>
    inline void CacheManager<T>::create_mask(thrust::device_vector<bool> &mask, CacheDevice<U> &cache)
    {
        // Get the LRU
        static thrust::device_vector<LruContent> &lru = cache.get_lru();

        // Get the LRU size
        uint32_t lruSize = static_cast<uint32_t>(lru.size());

        // The mask must have the same size as the LRU.
        mask.resize(lruSize, false);

        // Prepare the kernel launch
        const dim3 nbThreadsPerBlock(std::min(128U, lruSize), 1, 1);
        uint32_t numBlock = (lruSize % nbThreadsPerBlock.x != 0) ? 
            (lruSize / nbThreadsPerBlock.x + 1) :
            (lruSize / nbThreadsPerBlock.x);
        // NOTE : try in 2 dimensions instead of 1 (with a max limit (65536U ?) on the first dimension)
        const dim3 nbBlocks(numBlock, 1, 1);

        tdns::math::Vector3ui blockSize = cache.get_block_size();

        // Launch the kernel to fill the usage mask with flags to determine used and unused elements.
        create_usage_mask<<<nbBlocks, nbThreadsPerBlock>>>(
                make_uint3(blockSize[0], blockSize[1], blockSize[2]),
                lru.data().get(),
                lruSize,
                cache.get_usage_buffer().to_kernel_object(),
                _timestamp,
                mask.data().get());
#if TDNS_MODE == TDNS_MODE_DEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
        CUDA_CHECK_KERNEL_ERROR();
    // printf("-------------Mask complete-------------\n");
    }
} // namespace gpucache
} // namespace tdns