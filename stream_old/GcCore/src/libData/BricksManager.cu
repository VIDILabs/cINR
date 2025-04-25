#include <GcCore/libData/BricksManager.hpp>

#include <memory>
#include <fstream>
#include <iterator>

#include <lz4hc.h>
#include <lz4.h>

#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/Memory.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <fstream>

// Copied from cuda_utils.h for better functionality than just cudaDeviceSynchronize()
#define CUDA_SYNC_CHECK()                                                                             \
  do {                                                                                                \
    cudaDeviceSynchronize();                                                                          \
    cudaError_t error = cudaGetLastError();                                                           \
    if (error != cudaSuccess) {                                                                       \
      const char* msg = cudaGetErrorString(error);                                                    \
      fprintf(stderr, "CUDA sync error (%s: line %d): %s\n", __FILE__, __LINE__, msg);                \
      throw std::runtime_error(std::string("CUDA cudaDeviceSynchronize() failed with error ") + msg); \
    }                                                                                                 \
  } while (0)

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    BricksManager::BricksManager(const std::string &path, const tdns::math::Vector3ui &brickSize,
        const tdns::math::Vector3ui &bigBrickSize, uint32_t numberEncodedBytes, size_t cacheSize /* = 32768 */) :
    _brickEdgeSize(brickSize),
    _bigBrickSize(bigBrickSize),
    _numberEncodedBytes(numberEncodedBytes),
    _INRpath(path),
    _cache(cacheSize)
    {
        vnr::vec3f *h_coords;
        h_coords = new vnr::vec3f[_brickEdgeSize[0]*_brickEdgeSize[1]*_brickEdgeSize[2]];

        int n = 0;
        for (uint i=0; i<_brickEdgeSize[2]; ++i) {
            for (uint j=0; j<_brickEdgeSize[1]; ++j) {
                for (uint k=0; k<_brickEdgeSize[0]; ++k) {
                    h_coords[n] = vnr::vec3f(k,j,i);
                    n++;
                }
            }
        }
        const uint N = n * sizeof(vnr::vec3f);

        cudaMalloc(&_d_coords, N);
        cudaMemcpy(_d_coords, h_coords, N, cudaMemcpyHostToDevice);

        cudaMalloc(&_d_outCoords, n * sizeof(vnr::vec3f));
        cudaMalloc(&_d_values, n * sizeof(float) * 40);

        // cudaMalloc(&_d_outCoords, n * sizeof(vnr::vec3f) * 50); // hardcoded max num requests for both for now
        // cudaMalloc(&_d_values, n * sizeof(float) * 50);

        delete[] h_coords;

        std::cout << path << std::endl;

        vnrJson root = vnrCreateJsonBinary(path);

        _net = vnrCreateNetwork(root);

        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        uint32_t x,y,z;
        conf.get_field("size_X", x);
        conf.get_field("size_Y", y);
        conf.get_field("size_Z", z);
        _volDims = vnr::vec3f(x,y,z);

        std::cout << "Volume Dims: "<< _volDims.x << ' ' << _volDims.y << ' ' << _volDims.z << std::endl;

        // if (root.contains("volume")) {
        //     _volDims = vnr::vec3f(root["volume"]["dims"]["x"].get<int>(),
        //                           root["volume"]["dims"]["y"].get<int>(),
        //                           root["volume"]["dims"]["z"].get<int>());
            // _volDims = vnr::vec3f(4096,4096,4096);
            
            // Manual dims saving for OUT-OF-CORE trained params

            // root["volume"] = {
            //     { "dims", {
            //     { "x", 10240 },
            //     { "y", 7680 },
            //     { "z", 1536 }
            //     }}
        //     };

        //     const auto broot = vnrJson::to_bson(root);
        //     std::ofstream ofs("new_params.json", std::ios::binary | std::ios::out);
        //     ofs.write((char*)broot.data(), broot.size());
        //     ofs.close();

            // End of Manual saving

            // std::cout << "Volume Dims: "<< _volDims.x << ' ' << _volDims.y << ' ' << _volDims.z << std::endl;
        // }
        // else {
        //     std::cout << "No Volume Dims" << std::endl;
        //     root["volume"] = {
        //         { "dims", {
        //             { "x", 10240 },
        //             { "y", 7680 },
        //             { "z", 1536 }
        //         }}
        //     };
        // }

        // _net = vnrCreateNetwork(root);
    }

    //---------------------------------------------------------------------------------------------------
    Brick* BricksManager::get_brick(uint32_t level, const tdns::math::Vector3ui &position)
    {
        LOGDEBUG(10, tdns::common::log_details::Verbosity::INSANE, "Get brick Level [" << level
            << "] position [" << position[0] << " - " << position[1] << " - " << position[2] << "].");

        Bkey key = get_key(level, position);
        // check in cache
        auto it = _bricks.find(key);
        if (it != _bricks.end())
        {
            _cache.update(key);
            return it->second.get();
        }

        //load the brick from the HDD.
        return load_brick(key, level, position, 0);
    }

    //---------------------------------------------------------------------------------------------------
    __global__ void fillCoords(const vnr::vec3i brick_size, const vnr::vec3i brick_offset, const uint32_t lod_scale, const vnr::vec3f vol_dims, 
                               vnr::vec3f* __restrict__ d_coords, vnr::vec3f* __restrict__ d_outCoords) {
        
        vnr::vec3i id(threadIdx.x + blockIdx.x * blockDim.x,
                      threadIdx.y + blockIdx.y * blockDim.y,
                      threadIdx.z + blockIdx.z * blockDim.z);

        if (id.x >= brick_size.x) return;
        if (id.y >= brick_size.y) return;
        if (id.z >= brick_size.z) return;

        const uint32_t idx = (id.z * brick_size.x * brick_size.y) + (id.y * brick_size.x) + id.x;

        d_outCoords[idx].x = (brick_offset.x + d_coords[idx].x - 0.5) * lod_scale / vol_dims.x;
        d_outCoords[idx].y = (brick_offset.y + d_coords[idx].y - 0.5) * lod_scale / vol_dims.y;
        d_outCoords[idx].z = (brick_offset.z + d_coords[idx].z - 0.5) * lod_scale / vol_dims.z;

    }

    BricksManager::BrickStatus BricksManager::get_brick(uint32_t level, const tdns::math::Vector3ui &position, Brick **brick, uint32_t i)
    {
        LOGDEBUG(10, tdns::common::log_details::Verbosity::INSANE, "Get brick Level [" << level
        << "] position [" << position[0] << " - " << position[1] << " - " << position[2] << "].");

        // Bkey key = get_key(level, position);

        //Search in empty list
        // {
        //     auto it = _emptyBricks.find(key);
        //     if(it != _emptyBricks.end()) return BrickStatus::Empty;
        // }

        //check in cache
        // {
        //     auto it = _bricks.find(key);
        //     if (it != _bricks.end())
        //     {
        //         { // rustine
        //             std::lock_guard<std::mutex> guard(_lock);
        //             _cache.update(key);
        //         }
        //         *brick = it->second.get();
        //         return BrickStatus::Success;
        //     }
        // }

        //load the brick from the storage device.
        // *brick = load_brick(key, level, position, i);
        // return *brick ? BrickStatus::Success : BrickStatus::Unknown;

        const uint n = _brickEdgeSize[0] * _brickEdgeSize[1] * _brickEdgeSize[2];
        const vnr::vec3i brick_size(_brickEdgeSize[0], _brickEdgeSize[1], _brickEdgeSize[2]);
        const vnr::vec3i brick_offset(position[0] * (_brickEdgeSize[0] - 2),
                                      position[1] * (_brickEdgeSize[1] - 2),
                                      position[2] * (_brickEdgeSize[2] - 2));
        const uint32_t lod_scale = pow(2,level);


        dim3 blockDim(16, 8, 8);
        dim3 gridDim((_brickEdgeSize[0] + blockDim.x -1) / blockDim.x,
                     (_brickEdgeSize[1] + blockDim.y -1) / blockDim.y,
                     (_brickEdgeSize[2] + blockDim.z -1) / blockDim.z);
        // CUDA_SYNC_CHECK();
        fillCoords<<<gridDim,blockDim>>>(brick_size, brick_offset, lod_scale, _volDims, _d_coords, _d_outCoords);
        // CUDA_SYNC_CHECK();

        vnrInfer(_net, (vnrDevicePtr)(_d_outCoords), (vnrDevicePtr)(_d_values + (n*i)), n);

        return BrickStatus::Success;
    }

    void BricksManager::get_bricks(uint32_t* levels, tdns::math::Vector3ui* positions, Brick** bricks, BricksManager::BrickStatus* statusArr, const uint32_t count)
    {
        Bkey keys[count];

        for (size_t i=0; i < count; i++) 
        {
            keys[i] = get_key(levels[i], positions[i]);
            
            //Search in empty list (We will never have "empty" bricks)
            // {
            //     auto it = _emptyBricks.find(keys[i]);
            //     if(it != _emptyBricks.end()) 
            //         status[i] = BrickStatus::Empty;
            // }
        }


        load_bricks(keys, levels, positions, bricks, count);

        for (size_t i=0; i < count; i++) 
        {
            statusArr[i] = bricks[i] ? BrickStatus::Success : BrickStatus::Unknown;
        }
    }

    //---------------------------------------------------------------------------------------------------
    float BricksManager::write_brick(const std::string &outputDirectory,
        const Brick &brick,
        const tdns::math::Vector3ui &brickSize,
        bool compression /*= true*/)
    {
        std::string filePath = get_brick_path(outputDirectory, brickSize, brick.get_level(), brick.get_position());
        std::ofstream os(filePath, std::ios::out | std::ofstream::binary);

        tdns::math::Vector3ui brickEdgeSize = brick.get_edge_size();
        uint32_t nbBytes = brickEdgeSize[0] * brickEdgeSize[1] * brickEdgeSize[2] * brick.get_encoded();

        float ratio = 0.f;
        if(compression && nbBytes < LZ4_MAX_INPUT_SIZE) // LZ4_MAX_INPUT_SIZE = 2 113 929 216 bytes (almost 2Go) max size for compression 
        {
            // LZ4_compress_default() compress faster when dest buffer size is >= LZ4_compressBound(srcSize)
            uint64_t maxCompressedSize = LZ4_compressBound(nbBytes);
            std::vector<uint8_t> dataCompressed(maxCompressedSize);
            uint64_t compressedSize = LZ4_compress_HC(
                reinterpret_cast<const char*>(brick.get_data().data()),
                reinterpret_cast<char *>(dataCompressed.data()),
                nbBytes,
                static_cast<int>(maxCompressedSize),
                9);

            std::copy(dataCompressed.begin(), dataCompressed.begin() + compressedSize, std::ostreambuf_iterator<char>(os));
            ratio = (100.f - ((compressedSize * 100.f) / nbBytes));
        }
        else
        {
            std::copy(brick.get_data().begin(), brick.get_data().end(), std::ostreambuf_iterator<char>(os));
        }

        return ratio;
    }

    //---------------------------------------------------------------------------------------------------
    void BricksManager::load(const MetaData &metaData)
    {
        _bricks.clear();
        _emptyBricks.clear();
        _cache.clear();

        const std::vector<tdns::data::Bkey> &emptyBricks = metaData.get_empty_bricks();
        for(auto it = emptyBricks.begin(); it != emptyBricks.end(); ++it)
            _emptyBricks.insert(*it);
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_status_string(BricksManager::BrickStatus status) const
    {
        switch(status)
        {
            case BricksManager::BrickStatus::Success:
                return "Success";
            case BricksManager::BrickStatus::Unknown:
                return "Unknown";
            case BricksManager::BrickStatus::Empty:
                return "Empty";
            default:
                return "Error Status";
        }
    }

    //---------------------------------------------------------------------------------------------------
    void BricksManager::check_level_directory(const std::string &volumeDirectory, uint32_t level,
        const tdns::math::Vector3ui &brickSize)
    {
        std::string bricksDirectory = volumeDirectory + get_brick_folder(brickSize);
        if (!tdns::common::is_dir(bricksDirectory)) tdns::common::create_folder(bricksDirectory);

        std::string levelDirectory = bricksDirectory + "/L" + std::to_string(level);
        if (!tdns::common::is_dir(levelDirectory)) tdns::common::create_folder(levelDirectory);
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_brick_folder(const tdns::math::Vector3ui &brickSize)
    {
        return "bricks_" + std::to_string(brickSize[0]) + "_" + std::to_string(brickSize[1]) + "_" + std::to_string(brickSize[2]);
    }
    
    //---------------------------------------------------------------------------------------------------
    void BricksManager::insert_in_cache(const Bkey &key, Brick *brick)
    {
        auto oldestValue = _cache.push_back(key, brick);
        if (!oldestValue) return; //nothing else to do 
        
        auto it = _bricks.find(oldestValue->first);
        if (it != _bricks.end())
        {
            _bricks.erase(it);
        }
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_brick_path(const std::string &baseDirectory,
        const tdns::math::Vector3ui &brickSize,
        uint32_t level,
        const tdns::math::Vector3ui &position)
    {
        return baseDirectory + get_brick_folder(brickSize) + "/"
            + get_level_folder(level) + get_brick_name(level, position) + ".raw";
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_level_folder(uint32_t level)
    {
        return "L" + std::to_string(level) + "/";
    }

    //---------------------------------------------------------------------------------------------------

    // __global__ void fillCoords(const vnr::vec3i brick_size, const vnr::vec4i requested_bricks, const vnr::vec3f vol_dims, 
    //                            vnr::vec3f* __restrict__ d_coords, vnr::vec3f* __restrict__ d_outCoords) {
        
    //     vnr::vec3i id(threadIdx.x + blockIdx.x * blockDim.x,
    //                   threadIdx.y + blockIdx.y * blockDim.y,
    //                   threadIdx.z + blockIdx.z * blockDim.z);

    //     if (id.x >= brick_size.x) return;
    //     if (id.y >= brick_size.y) return;
    //     if (id.z >= brick_size.z) return;

    //     const vnr::vec3i brick_offset(requested_bricks.x * (brick_size.x - 2),
    //                                   requested_bricks.y * (brick_size.y - 2),
    //                                   requested_bricks.z * (brick_size.z - 2));
    //     const uint32_t lod_scale = pow(2,requested_bricks.w);

    //     const uint32_t idx = (id.z * brick_size.x * brick_size.y) + (id.y * brick_size.x) + id.x;

    //     d_outCoords[idx].x = (brick_offset.x + d_coords[idx].x - 0.5) * lod_scale / vol_dims.x;
    //     d_outCoords[idx].y = (brick_offset.y + d_coords[idx].y - 0.5) * lod_scale / vol_dims.y;
    //     d_outCoords[idx].z = (brick_offset.z + d_coords[idx].z - 0.5) * lod_scale / vol_dims.z;

    // }

    Brick* BricksManager::load_brick(Bkey key, uint32_t level, const tdns::math::Vector3ui &position, const uint32_t i)
    {
        if (_numberEncodedBytes != 4) {
            LOGERROR(10, "[" << _numberEncodedBytes << "] Encoded bytes detected. Only floats currently supported");
            return nullptr;
        }

        // const uint n = _brickEdgeSize[0] * _brickEdgeSize[1] * _brickEdgeSize[2];
        // const vnr::vec3i brick_size(_brickEdgeSize[0], _brickEdgeSize[1], _brickEdgeSize[2]);
        // const vnr::vec3i brick_offset(position[0] * (_brickEdgeSize[0] - 2),
        //                               position[1] * (_brickEdgeSize[1] - 2),
        //                               position[2] * (_brickEdgeSize[2] - 2));
        // const uint32_t lod_scale = pow(2,level);


        // dim3 blockDim(16, 8, 8);
        // dim3 gridDim((_brickEdgeSize[0] + blockDim.x -1) / blockDim.x,
        //              (_brickEdgeSize[1] + blockDim.y -1) / blockDim.y,
        //              (_brickEdgeSize[2] + blockDim.z -1) / blockDim.z);
        // // CUDA_SYNC_CHECK();
        // fillCoords<<<gridDim,blockDim>>>(brick_size, brick_offset, lod_scale, _volDims, _d_coords, _d_outCoords);
        // // CUDA_SYNC_CHECK();

        // vnrInfer(_net, (vnrDevicePtr)(_d_outCoords), (vnrDevicePtr)(_d_values + (n*i)), n);



        // CUDA_SYNC_CHECK();

        // std::vector<uint8_t> vnr_data(n * _numberEncodedBytes);

        // cudaMemcpy(vnr_data.data(), _d_values, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Fill brick data and update LRU
        // std::unique_ptr<Brick> brick = tdns::common::create_unique_ptr<Brick>(_brickEdgeSize * _bigBrickSize, _numberEncodedBytes);

        // brick->set_data(vnr_data);
        // brick->set_level(level);
        // brick->set_position(position);

        // Brick *ptr = brick.get();
        Brick *ptr = nullptr;
        // {
        //     std::lock_guard<std::mutex> guard(_lock);
        //     _bricks[key].swap(brick);
        //     insert_in_cache(key, ptr);
        // }

        return ptr;
    }

    __global__ void fillAllCoords(const vnr::vec3i brick_size, vnr::vec3i* brick_offsets, uint32_t* lod_scales, const vnr::vec3f vol_dims, const uint32_t count,
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

        // Calculate the linear index for the specific coordinate
        const uint32_t idx = (id.z * brick_size.x * brick_size.y) + (id.y * brick_size.x) + id.x;
        const uint32_t outIndex = idx + (brick_size.x * brick_size.y * brick_size.z * brick_id);

        d_outCoords[outIndex].x = (brick_offsets[brick_id].x + d_coords[idx].x - 0.5) * lod_scales[brick_id] / vol_dims.x;
        d_outCoords[outIndex].y = (brick_offsets[brick_id].y + d_coords[idx].y - 0.5) * lod_scales[brick_id] / vol_dims.y;
        d_outCoords[outIndex].z = (brick_offsets[brick_id].z + d_coords[idx].z - 0.5) * lod_scales[brick_id] / vol_dims.z;

        // printf("lod: %u\n", lod_scales[brick_id]);
    }

    void BricksManager::load_bricks(Bkey* keys, uint32_t* levels, tdns::math::Vector3ui* positions, Brick** bricks, const uint32_t count)
    {
        if (_numberEncodedBytes != 4) {
            LOGERROR(10, "[" << _numberEncodedBytes << "] Encoded bytes detected. Only floats currently supported");
            return;
        }

        const uint n = _brickEdgeSize[0] * _brickEdgeSize[1] * _brickEdgeSize[2];
        const vnr::vec3i brick_size(_brickEdgeSize[0], _brickEdgeSize[1], _brickEdgeSize[2]);

        vnr::vec3i brick_offsets[count];
        uint32_t lod_scales[count];

        for (size_t i=0; i < count; i++) 
        {
            brick_offsets[i] = vnr::vec3i(positions[i][0] * (_brickEdgeSize[0] - 2),
                                          positions[i][1] * (_brickEdgeSize[1] - 2),
                                          positions[i][2] * (_brickEdgeSize[2] - 2));
            lod_scales[i] = pow(2, levels[i]);
        }

        vnr::vec3i *d_brick_offsets;
        uint32_t *d_lod_scales;

        cudaMalloc(&d_brick_offsets, count * sizeof(vnr::vec3i));
        cudaMalloc(&d_lod_scales, count * sizeof(uint32_t));

        cudaMemcpy(d_brick_offsets, brick_offsets, count * sizeof(vnr::vec3i), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lod_scales, lod_scales, count * sizeof(uint32_t), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 8, 8);
        dim3 gridDim(((_brickEdgeSize[0] * count) + blockDim.x -1) / blockDim.x,
                     ((_brickEdgeSize[1] * count) + blockDim.y -1) / blockDim.y,
                     ((_brickEdgeSize[2] * count) + blockDim.z -1) / blockDim.z);

        fillAllCoords<<<gridDim,blockDim>>>(brick_size, d_brick_offsets, d_lod_scales, _volDims, count, _d_coords, _d_outCoords);
        CUDA_SYNC_CHECK();

        vnrInfer(_net, (vnrDevicePtr)(_d_outCoords), (vnrDevicePtr)(_d_values), n * count);

        std::vector<uint8_t> vnr_data(n * count * _numberEncodedBytes);

        cudaMemcpy(vnr_data.data(), _d_values, n * count * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Fill brick data and update LRU
        // Brick* ret[count];
        for (uint32_t i=0; i < count; i++)
        {
            std::unique_ptr<Brick> brick = tdns::common::create_unique_ptr<Brick>(_brickEdgeSize * _bigBrickSize, _numberEncodedBytes);

            brick->set_data(vnr_data, i * n * _numberEncodedBytes, n * _numberEncodedBytes);
            brick->set_level(levels[i]);
            brick->set_position(positions[i]);

            Brick *ptr = brick.get();
            {
                std::lock_guard<std::mutex> guard(_lock);
                _bricks[keys[i]].swap(brick);
                insert_in_cache(keys[i], ptr);
            }
            bricks[i] = ptr;
        }
        cudaFree(d_brick_offsets);
        cudaFree(d_lod_scales);
    }
} //namespace data
} //namespace tdns