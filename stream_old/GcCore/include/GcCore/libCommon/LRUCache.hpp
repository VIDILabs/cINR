#pragma once

#include <unordered_map>
#include <memory>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Least Recently Used cache.
    * The last used pair is at the end of the cache and the least
    * recently used is at the beginning of it.
    * /!\ Be carefull when you loop over it. Almost all operations modify
    * the container.
    * /!\ This container uses the erase / copy to update a pair,
    * prefere the use of pointer in the values.
    *
    * @template Key of the pair. It MUST implement the operator == and MUST BE copyable.
    * @template Value of the pair. It MUST BE copyable.
    */
    template <typename Key, typename Value>
    class LRUCache
    {
    public:
        typedef typename std::unordered_map<Key, Value>::iterator iterator;
        typedef typename std::unordered_map<Key, Value>::const_iterator const_iterator;

    public:

        /**
        * @brief Constructor.
        *
        * @param the maximum size of the cache.
        */
        LRUCache(size_t maxSize)
        {
            _maxSize = maxSize;
        }

        /**
        * @brief Iterator begin and end.
        */
        iterator begin() { return _values.begin(); }
        const_iterator begin() const { return _values.cbegin(); }

        iterator end() { return _values.end(); }
        const_iterator end() const { return _values.cend(); }

        /**
        * @brief Add a new pair in the cache.
        * If the pair already exist it will remove it and add the one given in parameter.
        * If the cache is full it will erase the fist (begin) pair and at the new one at the end.
        *
        * @param Key of the pair.
        * @param Value of the pair.
        *
        * @retrun A unique pointer of the removed pair. If the pointer is null it means the last element has not
        * been deleted, else the pointer contains the removed pair.
        */
        std::unique_ptr<std::pair<Key, Value>> push_back(const Key &key, const Value &value)
        {
            std::unique_ptr<std::pair<Key, Value>> ptr = nullptr;
            iterator it = _values.find(key);
            if (it != _values.end()) //already exists.
            {
                _values.erase(it);
            }
            else if (_values.size() >= _maxSize) //cache is full.
            {
                iterator old = _values.begin();
                ptr = create_unique_ptr<std::pair<Key, Value>>(std::make_pair(old->first, old->second));
                _values.erase(old);
            }

            _values.insert({ key, value });
            return ptr;
        }

        /**
        * @brief Search for a value. If the value exist it will autmatically place it at the
        * beginning of the cache.
        *
        * @param Key of the pair to find.
        * @return Iterator on the pair if found, end() otherwise.
        */
        iterator find(const Key &key)
        {
            iterator it = _values.find(key);
            if (it == _values.end()) return it;

            std::pair<Key, Value> pair = *it;
            _values.erase(it);
            _values.insert(pair);
            return _values.end();
        }

        /**
        * @brief Remove an element from the cache.
        *
        * @param The element to remove.
        * @return The next element after the removed element.
        */
        iterator erase(const_iterator pair)
        {
            return _values.erase(pair);
        }

        /**
        * @brief Update the pair and put it at the top of the cache.
        *
        * @param Key of the pair to update.
        */
        void update(const Key &key)
        {
            iterator it = _values.find(key);
            if (it == _values.end()) return;

            std::pair<Key, Value> pair = *it;
            _values.erase(it);
            _values.insert(pair);
        }

        /**
        * @brief Clear the lru.
        */
        void clear()
        {
            _values.clear();
        }

    private:
        /**
        * Member data
        */
        size_t                          _maxSize;   ///< Maximum size the cache can handle.
        std::unordered_map<Key, Value>  _values;    ///< All pairs key - value.
    };
} // namespace common
} // namespace tdns