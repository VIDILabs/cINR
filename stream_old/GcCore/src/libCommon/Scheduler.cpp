#include <GcCore/libCommon/Scheduler.hpp>
#include <iostream>

#include <GcCore/libCommon/ThreadPool.hpp>

namespace tdns
{
namespace common
{
    //---------------------------------------------------------------------------------------------------
    void Scheduler::register_task(AbstractTask *task)
    {  
        register_task(std::bind(&AbstractTask::do_task, task));
    }
    
    //---------------------------------------------------------------------------------------------------
    void Scheduler::register_task(std::function<void()> task)
    {
        ThreadPool::get_instance().queue(task);
    }

    //---------------------------------------------------------------------------------------------------
    void Scheduler::register_periodic_task(AbstractTask *task /* delta T*/)
    {        
        _taskList.push_back(
            std::pair<AbstractTask*, std::chrono::high_resolution_clock::time_point>(
                task, std::chrono::high_resolution_clock::now()
            )
        );
    }

    //---------------------------------------------------------------------------------------------------
    void Scheduler::schedule()
    {
        _run = true;

        std::chrono::high_resolution_clock::time_point now;
        uint64_t deltaTime;

        uint64_t minTime, wakeupTime;
        bool waitingTask;

        while (_run)
        {
            waitingTask = false;
            minTime = 8000; // 8 second

            // Foreach list
            for (std::pair<AbstractTask*, std::chrono::high_resolution_clock::time_point> last_task : _taskList)
            {
                now = std::chrono::high_resolution_clock::now();
                deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_task.second).count();
               
                // If delta time > task->deltaTime 
                if (deltaTime > last_task.first->deltaTime())
                {
                    register_task(std::bind(&AbstractTask::do_task, last_task.first));
                    last_task.second = now;
                }
                else
                {
                    waitingTask = true;

                    uint64_t t = (deltaTime - last_task.first->deltaTime());
                    minTime = minTime < t ? minTime : t;
                }
            }

            // How many time sleep ?
            if (!waitingTask)
                wakeupTime = 500;
            else
                wakeupTime = minTime;
            
            // sleep a moment
            std::this_thread::sleep_for(std::chrono::milliseconds(wakeupTime));
        }
    }

    //---------------------------------------------------------------------------------------------------
    void Scheduler::stop()
    {
        _run = false;
    }

} // common
} // tdns