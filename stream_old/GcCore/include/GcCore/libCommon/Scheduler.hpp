#pragma once

#include <list>
#include <chrono>
#include <functional>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/AbstractTask.hpp>
#include <GcCore/libCommon/Singleton.hpp>

namespace tdns
{
namespace common
{
    class TDNS_API Scheduler : public Singleton<Scheduler>
    {
    public:
        /**
        * @brief register a task
        *
        * @param[in]    task    task to register
        */
        void register_task(AbstractTask *task);
        void register_task(std::function<void()> task);

        /**
        * @brief register a periodic task
        *
        * @param[in]    task    task to register
        */
        void register_periodic_task(AbstractTask *task);

        /**
        * @brief schedule
        */
        void schedule();

        /**
        * @brief stop the scheduler (and stop thread pool)
        */
        void stop();
    protected:
         /*
         * Data member
         */
        std::list<std::pair<AbstractTask*, std::chrono::high_resolution_clock::time_point> > _taskList;  ///< List of periodic task
        bool                                                                                 _run;       ///< running

    };
} // common
} // tdns