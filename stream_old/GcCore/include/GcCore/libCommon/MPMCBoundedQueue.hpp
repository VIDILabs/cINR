#pragma once

#include <atomic>
#include <cassert>

namespace tdns
{
namespace common
{
    /**
    * @brief A lock free implementation of a multi producers, multi consumers queue.
    *
    * This class implements a high-performance bounded concurrent queue that 
    * supports multiple producers, multiple consumers, and optional blocking.
    * The queue has a fixed capacity which is fixed in the constructor.
    * 
    * @tparam [T] The element type the MPMCBoundedQueue will apply.
    * @warning [T] MUST be copyable.
    * @see More information: http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
    */
    template<typename T>
    class MPMCBoundedQueue
    {
    public:
        /**
        * @brief Default constructor.
        *
        * Constructor the queue with a size of 1024 elements.
        * 
        * @param[in]   buffer_size Queue maximum size. Default value is 1024.
        *
        * @note For performance reason, it should be a power of 2.
        */
        MPMCBoundedQueue(size_t buffer_size = 1024);

        /**
        * @brief Destructor
        *
        * Erase all pending element from memory.
        */
        ~MPMCBoundedQueue();

        /**
        * @brief Add an element to the queue.
        * Insert a copy the element at the end of the queue.
        *
        * @param[in]    data    The element to insert.
        *
        * @return True on success, false otherwise.
        */
        bool push(T const &data);

        /**
        * @brief Remove an element to the queue.
        * Remove the front element of the queue.
        *
        * @param[out]   data   The element which has been removed.
        *
        * @return True on success, false otherwise.
        */
        bool pop(T& data);

        /**
        * @brief Tell if the queue is empty or not.
        *
        * @return True if empty, false otherwise.
        */
        bool empty() const;

    private:
        /**
        * @brief Disable constructor by copy.
        */
        MPMCBoundedQueue(MPMCBoundedQueue const &) = delete;

        /**
        * @brief Disable equal operator.
        */
        MPMCBoundedQueue& operator = (MPMCBoundedQueue const &) = delete;

    private:
        /**
        * @brief This structure represents the data internally used by the queue.
        * The idea is to get everything as atomic as possible. Hence,
        * we will mainly work with the sequence member of this struct.
        */
        struct cell_t
        {
            std::atomic<size_t>   _cell_sequence;
            T                     _data;
        };

        static size_t const     cacheline_size = 64;
        typedef char            cacheline_pad_t[cacheline_size];
        //We use the cachelinepadding trick to avoid false sharing at CPU level.
        cacheline_pad_t         _pad0;
        cell_t* const           _buffer;        ///< The buffer of data. Used like a ring buffer.
        size_t const            _buffer_mask;   ///< Used to compute the position in the buffer.
        cacheline_pad_t         _pad1;
        std::atomic<size_t>     _enqueue_pos;   ///< Position to enqueue.
        cacheline_pad_t         _pad2;
        std::atomic<size_t>     _dequeue_pos;   ///< Position to dequeue.
        cacheline_pad_t         _pad3;
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline MPMCBoundedQueue<T>::MPMCBoundedQueue(size_t buffer_size /* = 1024 */) : 
        _buffer(new cell_t[buffer_size]),
        _buffer_mask(buffer_size - 1)
    {
        assert((buffer_size >= 2) && ((buffer_size & (buffer_size - 1)) == 0));
        for (size_t i = 0; i != buffer_size; i += 1)
            _buffer[i]._cell_sequence.store(i, std::memory_order_relaxed);

        _enqueue_pos.store(0, std::memory_order_relaxed);
        _dequeue_pos.store(0, std::memory_order_relaxed);
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline MPMCBoundedQueue<T>::~MPMCBoundedQueue()
    {
        delete[] _buffer;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline bool MPMCBoundedQueue<T>::push(T const &data)
    {
        cell_t *cell;
        size_t pos = _enqueue_pos.load(std::memory_order_relaxed);
        for (;;)
        {
            cell = &_buffer[pos & _buffer_mask];
            size_t seq = cell->_cell_sequence.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)pos;
            if (dif == 0)
            {
                if (_enqueue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            }
            else if (dif < 0)
                return false;
            else
                pos = _enqueue_pos.load(std::memory_order_relaxed);
        }
        cell->_data = data;
        cell->_cell_sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline bool MPMCBoundedQueue<T>::pop(T &data)
    {
        cell_t* cell;
        size_t pos = _dequeue_pos.load(std::memory_order_relaxed);
        for (;;)
        {
            cell = &_buffer[pos & _buffer_mask];
            size_t seq = cell->_cell_sequence.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
            if (dif == 0)
            {
                if (_dequeue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            }
            else if (dif < 0)
                return false;
            else
                pos = _dequeue_pos.load(std::memory_order_relaxed);
        }
        data = cell->_data;
        cell->_cell_sequence.store(pos + _buffer_mask + 1, std::memory_order_release);
        return true;
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    inline bool MPMCBoundedQueue<T>::empty() const
    {
        return _dequeue_pos.load() == _enqueue_pos.load();
    }
} // namespace common
} // namespace tdns