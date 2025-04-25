#include <GcCore/libCommon/ThreadPool.hpp>

#include <iostream>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------
    ThreadPool::ThreadPool(size_t size /* = 4 */) : _run(true)
    {
        _pool.resize(size);

        for (size_t i = 0; i < size; ++i)
            _pool[i] = std::thread(&ThreadPool::run, this);   
    }

    //---------------------------------------------------------------------------------------------
    ThreadPool::~ThreadPool()
    {
        _run = false;
        // Notify all thread
        _cond_var.notify_all();

        // Join all thread
        for (std::thread &thread : _pool)
            thread.join();
    }

    //---------------------------------------------------------------------------------------------
    void ThreadPool::run()
    {
        std::function<void()> task;
        while (_run)
        {
            {
                std::unique_lock<std::mutex> lock(_mutex);
                while (_run && _tasks.empty())
                {
                    // Condition variable
                    _cond_var.wait(lock);
                }
            }

            if (!_tasks.pop(task)) continue;
            
            // Execute task
            task();
        }
    }

    //---------------------------------------------------------------------------------------------
    void ThreadPool::queue(std::function<void()> task)
    {
        if (_tasks.push(task))
            _cond_var.notify_one();
    }
} // common
} // tdns