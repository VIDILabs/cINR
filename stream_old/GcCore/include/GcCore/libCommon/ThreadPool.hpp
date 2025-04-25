#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/MPMCBoundedQueue.hpp>
#include <GcCore/libCommon/AbstractTask.hpp>
#include <GcCore/libCommon/Singleton.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief pool of thread
    */
    class TDNS_API ThreadPool : public Singleton<ThreadPool>
    {
    public:
        /**
        * @brief constructor
        *
        * @param    size    size of pool
        */
        ThreadPool(size_t size = 4);

        /**
        * @brief destructor
        */
        ~ThreadPool();
        
        /**
        * @brief thread method
        */
        void run();

        /**
        * @brief Add a task
        *
        * @param    task    the task
        * @param    delta   the period time
        */
        void queue(std::function<void()> task);

    protected:
        /*
        * Member data.
        */
        std::vector<std::thread>                _pool;      ///< Pool of thread
        MPMCBoundedQueue<std::function<void()>> _tasks;     ///< Task to execute
        bool                                    _run;       ///< Thread conditionnal
        std::condition_variable                 _cond_var;  ///< Conditionnal variable to sleep thread
        std::mutex                              _mutex;     ///< Mutex for conditionnal variable
    };
} // common
} // tdns