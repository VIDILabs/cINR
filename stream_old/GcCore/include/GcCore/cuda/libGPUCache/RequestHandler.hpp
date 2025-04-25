#pragma once

#include <cstring>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/cuda/libCommon/DynamicArray3dHost.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/Memory.hpp>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Operator.hpp>
#include <GcCore/libData/BricksManager.hpp>
#include <GcCore/libData/Brick.hpp>
#include <GcCore/libData/VolumeConfiguration.hpp>

namespace tdns
{
namespace gpucache
{
    template<typename T>
    class RequestHandler
    {
    public:

        /**
        * @brief
        */
        RequestHandler(const tdns::data::VolumeConfiguration &volumeConfiguration, size_t nbMaxRequests, int32_t gpuID = 0);

        void start();
        
        void stop();

        void notify_request(const thrust::device_vector<uint4> &askedBricks, std::function<void()> callback);

        void run();

        bool is_working() const;
        
        /**
        * @brief
        */
        const tdns::math::Vector3ui& get_brick_size() const;

        /**
        * @brief
        */
        const tdns::math::Vector3ui& get_big_brick_size() const;

        /**
        * @brief
        */
        const tdns::common::DynamicArray3dHost<T, tdns::common::DynamicArrayOptions::Options::Mapped>&
        get_request_buffer() const;

        tdns::common::DynamicArray3dHost<T, tdns::common::DynamicArrayOptions::Options::Mapped>&
        get_request_buffer();

        thrust::device_vector<uint4>& get_asked_bricks();

        const thrust::device_vector<uint4>& get_asked_bricks() const;

        thrust::device_vector<uint32_t>& get_big_brick_indexes();

        const thrust::device_vector<uint32_t>& get_big_brick_indexes() const;

        thrust::device_vector<uint4>& get_big_brick_coords();

        const thrust::device_vector<uint4>& get_big_brick_coords() const;
        
        size_t get_nb_non_empty_bricks() const;

        /**
        * @brief
        */
        bool handle_request(const thrust::host_vector<uint4> &requestedBricks);

    public:

        tdns::math::Vector3ui                                           _brickSize;         ///<
        tdns::math::Vector3ui                                           _bigBrickSize;      ///<
        std::unique_ptr<tdns::data::BricksManager>                      _bricksManager;     ///<
        std::unique_ptr<tdns::common::DynamicArray3dHost
            <T, tdns::common::DynamicArrayOptions::Options::Mapped>>    _requestBuffer;     ///<
        size_t                                                          _nbMaxRequests;     ///<
        thrust::device_vector<bool>                                     _emptyFlags;        ///<
        thrust::device_vector<uint32_t>                                 _bigBrickIndexes;   ///< Index of the big bricks in the mapped buffer for each small brick.
        thrust::device_vector<uint4>                                    _bigBrickCoords;    ///<
        size_t                                                          _nbNonEmptyBricks;  ///<
        thrust::device_vector<uint4>                                    _askedBricks;       ///<
        std::thread                                                     _thread;            ///<
        std::mutex                                                      _cvLock;            ///< Mutex only used to notify the thread new request.
        std::condition_variable                                         _cv;                ///<
        bool                                                            _run;               ///<
        bool                                                            _newRequest;        ///<
        std::function<void()>                                           _callback;          ///<
        int32_t                                                         _gpuID;             ///<
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline RequestHandler<T>::RequestHandler(const tdns::data::VolumeConfiguration &volumeConfiguration, size_t nbMaxRequests, int32_t gpuID /*= 0*/)
    {
        _bricksManager = tdns::common::create_unique_ptr<tdns::data::BricksManager>
            (volumeConfiguration.INR_Path, volumeConfiguration.BrickSize, volumeConfiguration.BigBrickSize, sizeof(T));
        _nbMaxRequests = nbMaxRequests;
        _requestBuffer = tdns::common::create_unique_ptr<tdns::common::DynamicArray3dHost
            <T, tdns::common::DynamicArrayOptions::Options::Mapped>>
            (tdns::math::Vector3ui(_nbMaxRequests *
                volumeConfiguration.BrickSize[0] * volumeConfiguration.BigBrickSize[0],
                volumeConfiguration.BrickSize[1] * volumeConfiguration.BigBrickSize[1],
                volumeConfiguration.BrickSize[2] * volumeConfiguration.BigBrickSize[2]));
        _bigBrickIndexes.resize(_nbMaxRequests * volumeConfiguration.BigBrickSize[0] * volumeConfiguration.BigBrickSize[1] * volumeConfiguration.BigBrickSize[2]);
        _bigBrickCoords.resize(_nbMaxRequests);
        _brickSize = volumeConfiguration.BrickSize;
        _bigBrickSize = volumeConfiguration.BigBrickSize;
        _newRequest = false;
        _gpuID = gpuID;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void RequestHandler<T>::start()
    {
        _run = true;
        _newRequest = false;
        _thread = std::thread(&RequestHandler::run, this);
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void RequestHandler<T>::stop()
    {
        if (_thread.joinable())
        {
            {
                std::lock_guard<std::mutex> guard(_cvLock);
                _askedBricks.clear();
                _run = false;
                _cv.notify_one(); // to release the thread from the _cv.wait.
            }
            _thread.join();
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void RequestHandler<T>::notify_request(const thrust::device_vector<uint4> &askedBricks, std::function<void()> callback)
    {
        // guard the CV
        std::lock_guard<std::mutex> guard(_cvLock);
        _askedBricks = askedBricks;
        _emptyFlags.resize(askedBricks.size());
        _newRequest = true;
        _callback = callback;
        _cv.notify_one();
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline void RequestHandler<T>::run()
    {
        CUDA_SAFE_CALL(cudaSetDevice(_gpuID));
        while (_run)
        {
            std::unique_lock<std::mutex> lock(_cvLock);
            while (_run && !_newRequest) // Guard for spurious wakeups
            {
                _cv.wait(lock);
            }

            if (_askedBricks.size() == 0) continue;

            thrust::host_vector<uint4> h_bricks = _askedBricks;
            handle_request(h_bricks);
            _callback();
            _newRequest = false;
        }
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline bool RequestHandler<T>::is_working() const
    {
        return _newRequest;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::math::Vector3ui& RequestHandler<T>::get_brick_size() const
    {
        return _brickSize;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::math::Vector3ui& RequestHandler<T>::get_big_brick_size() const
    {
        return _bigBrickSize;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const tdns::common::DynamicArray3dHost<T, tdns::common::DynamicArrayOptions::Options::Mapped>&
    RequestHandler<T>::get_request_buffer() const
    {
        return *_requestBuffer;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline tdns::common::DynamicArray3dHost<T, tdns::common::DynamicArrayOptions::Options::Mapped>&
    RequestHandler<T>::get_request_buffer()
    {
        return *_requestBuffer;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline thrust::device_vector<uint4>& RequestHandler<T>::get_asked_bricks()
    {
        return _askedBricks;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const thrust::device_vector<uint4>& RequestHandler<T>::get_asked_bricks() const
    {
        return _askedBricks;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline thrust::device_vector<uint32_t>& RequestHandler<T>::get_big_brick_indexes()
    {
        return _bigBrickIndexes;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const thrust::device_vector<uint32_t>& RequestHandler<T>::get_big_brick_indexes() const
    {
        return _bigBrickIndexes;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline thrust::device_vector<uint4>& RequestHandler<T>::get_big_brick_coords()
    {
        return _bigBrickCoords;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline const thrust::device_vector<uint4>& RequestHandler<T>::get_big_brick_coords() const
    {
        return _bigBrickCoords;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline size_t RequestHandler<T>::get_nb_non_empty_bricks() const
    {
        return _nbNonEmptyBricks;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline bool RequestHandler<T>::handle_request(const thrust::host_vector<uint4> &requestedBricks)
    {
        _nbNonEmptyBricks = 0;
        uint32_t nbBigBrickAdded = 0;
        tdns::math::Vector3ui bigBrickSize = _bigBrickSize;
        uint32_t nbSmallBricksInBigOne = bigBrickSize[0] * bigBrickSize[1] * bigBrickSize[2];
        size_t bigBrickSizeVoxels = _brickSize[0] * _brickSize[1] * _brickSize[2] * nbSmallBricksInBigOne;
        tdns::data::Brick *brick = nullptr;
        tdns::data::BricksManager &bricksManager = *_bricksManager;

        static std::map<uint4, int32_t> bricksAlreadyAsked;
        uint32_t cptMap = 0;

        // batch test code start
        // const uint32_t nbReuqested = requestedBricks.size();

        // tdns::math::Vector3ui coordinates[nbReuqested];
        // uint32_t levels[nbReuqested];
        // tdns::data::Brick* bricks[nbReuqested];
        // for (size_t i = 0; i < nbReuqested; ++i)
        // {
        //     coordinates[i] = tdns::math::Vector3ui(requestedBricks[i].x / bigBrickSize[0],
        //                                            requestedBricks[i].y / bigBrickSize[1],
        //                                            requestedBricks[i].z / bigBrickSize[2]);
        //     levels[i] = requestedBricks[i].w;

        //     bricks[i] = nullptr;
            
        //     auto it = bricksAlreadyAsked.find({coordinates[i][0], coordinates[i][1], coordinates[i][2], levels[i]});
        //     if (it != bricksAlreadyAsked.end())
        //     {
        //         if(it->second == -1 )//empty brick
        //             _emptyFlags[i] = true;
        //         else // non empty brick
        //         {
        //             _emptyFlags[i] = false;
        //             _bigBrickIndexes[_nbNonEmptyBricks] = it->second;
        //             _nbNonEmptyBricks++;
        //         }
        //     }
        // }

        // tdns::data::BricksManager::BrickStatus statusArr[nbReuqested];
        // bricksManager.get_bricks(levels, coordinates, bricks, statusArr, nbReuqested);
        
        // for (size_t i = 0; i < nbReuqested; ++i)
        // {
        //     switch(statusArr[i])
        //     {
        //         case tdns::data::BricksManager::BrickStatus::Success:
        //         {
        //             //load brick in the mapped buffer
        //             std::memcpy(&(*_requestBuffer)[nbBigBrickAdded * bigBrickSizeVoxels], bricks[i]->get_data().data(), bigBrickSizeVoxels * sizeof(T));
        //             ++nbBigBrickAdded;
        //             //add it to the map as non empty brick
        //             bricksAlreadyAsked.insert({ {coordinates[i][0], coordinates[i][1], coordinates[i][2], levels[i]}, cptMap });
        //             if (_nbNonEmptyBricks > _nbMaxRequests)
        //                 assert(false && "Bricks number exceed the request buffer size ! RequestHandler.hpp handle_request()");
        //             _bigBrickIndexes[_nbNonEmptyBricks] = cptMap;
        //             _bigBrickCoords[cptMap] = {coordinates[i][0], coordinates[i][1], coordinates[i][2], levels[i]};
        //             cptMap++;

        //             _emptyFlags[i] = false;
        //             ++_nbNonEmptyBricks;
        //         }
        //         break;
        //         case tdns::data::BricksManager::BrickStatus::Empty:
        //         {
        //             //add it to the map as non empty brick
        //             bricksAlreadyAsked.insert({ {coordinates[i][0], coordinates[i][1], coordinates[i][2], levels[i]}, -1 });
        //             _emptyFlags[i] = true;
        //         }
        //         break;
        //         default:
        //             assert(false && "Unknown brick status ! We need to handle this !!!!!! RequestHandler.hpp handle_request()"); //We need to handle this !!!!!!
        //             continue;
        //     }
        // }

        // batch test code end

        for (uint32_t i = 0; i < requestedBricks.size(); ++i)
        {
            const uint4 &coordinate = requestedBricks[i];
            const uint4 &coordinateBigBrick = make_uint4(   coordinate.x / bigBrickSize[0],
                                                            coordinate.y / bigBrickSize[1],
                                                            coordinate.z / bigBrickSize[2],
                                                            coordinate.w);
            // auto it = bricksAlreadyAsked.find(coordinateBigBrick);
            // if (it != bricksAlreadyAsked.end())
            // {
            //     if(it->second == -1 )//empty brick
            //         _emptyFlags[i] = true;
            //     else // non empty brick
            //     {
            //         _emptyFlags[i] = false;
            //         _bigBrickIndexes[_nbNonEmptyBricks] = it->second;
            //         _nbNonEmptyBricks++;
            //     }
            //     continue;
            // }
            
            //load the brick
            tdns::data::BricksManager::BrickStatus status =
                bricksManager.get_brick(coordinateBigBrick.w,
                { coordinateBigBrick.x, coordinateBigBrick.y, coordinateBigBrick.z }, &brick, i);

            LOGINFO(40, tdns::common::log_details::Verbosity::INSANE, "Get brick Level [" << coordinateBigBrick.w
                << "] position [" << coordinateBigBrick.x << " - " << coordinateBigBrick.y << " - " << coordinateBigBrick.z 
                << "] status [" << bricksManager.get_status_string(status) << "]");

            switch(status)
            {
                case tdns::data::BricksManager::BrickStatus::Success:
                {
                    //load brick in the mapped buffer
                    // std::memcpy(&(*_requestBuffer)[nbBigBrickAdded * bigBrickSizeVoxels], brick->get_data().data(), bigBrickSizeVoxels * sizeof(T));
                    ++nbBigBrickAdded;
                    //add it to the map as non empty brick
                    bricksAlreadyAsked.insert({ coordinateBigBrick, cptMap });
                    if (_nbNonEmptyBricks > _nbMaxRequests)
                        assert(false && "Bricks number exceed the request buffer size ! RequestHandler.hpp handle_request()");
                    _bigBrickIndexes[_nbNonEmptyBricks] = cptMap;
                    _bigBrickCoords[cptMap] = coordinateBigBrick;
                    cptMap++;

                    _emptyFlags[i] = false;
                    ++_nbNonEmptyBricks;
                }
                break;
                case tdns::data::BricksManager::BrickStatus::Empty:
                {
                    //add it to the map as non empty brick
                    bricksAlreadyAsked.insert({ coordinateBigBrick, -1 });
                    _emptyFlags[i] = true;
                }
                break;
                default:
                    assert(false && "Unknown brick status ! We need to handle this !!!!!! RequestHandler.hpp handle_request()"); //We need to handle this !!!!!!
                    continue;
            }
        }
        // ------ Loop End ------

        if (_nbNonEmptyBricks - _askedBricks.size() != 0)
        {
            //-- sort non empty - empty
            thrust::device_vector<uint4> result(_askedBricks.size());
            
            thrust::device_vector<uint4>::iterator it = thrust::copy_if(
                _askedBricks.begin(),
                _askedBricks.end(),
                _emptyFlags.begin(),
                result.begin(),
                predicate::equal<bool>(false));

            thrust::copy_if(_askedBricks.begin(), _askedBricks.end(), _emptyFlags.begin(), it, predicate::equal<bool>(true));

            _askedBricks = result;
        }

        return true;
    }
} // namespace gpucache
} // namespace tdns