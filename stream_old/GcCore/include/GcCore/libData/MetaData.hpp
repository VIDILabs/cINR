/*
 */
#pragma once

#include <cstdint>
#include <vector>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libData/BrickKey.hpp>

namespace tdns
{
namespace data
{
    class TDNS_API MetaData
    {
    public:

        /**
        * @brief Constructor.
        */
        MetaData();

        /**
        * @brief Destructor.
        */
        ~MetaData() = default;

        /**
        * @brief Get the 3D initials dimensions of a level of resolution of the mipmap pyramid
        *
        * @param The desired level of resolution.
        * @return The desired level of resolution 3D initials dimensions
        */
        tdns::math::Vector3ui& get_initial_size(const size_t i) { return _initialLevels[i]; }
        const tdns::math::Vector3ui& get_initial_size(const size_t i) const { return _initialLevels[i]; }  
        
        /**
        * @brief Get the 3D initials dimensions of all the levels of resolution of the mipmap pyramid
        *
        * @return A vector with all the levels of resolution 3D initials dimensions
        */
        std::vector<tdns::math::Vector3ui>& get_initial_levels() { return _initialLevels; }
        const std::vector<tdns::math::Vector3ui>& get_initial_levels() const { return _initialLevels; }        

        /**
        * @brief Get the 3D reals dimensions of a level of resolution of the mipmap pyramid
        *
        * @param The desired level of resolution.
        * @return The desired level of resolution 3D reals dimensions
        */
        tdns::math::Vector3ui& get_real_size(const size_t i) { return _realLevels[i]; }
        const tdns::math::Vector3ui& get_real_size(const size_t i) const { return _realLevels[i]; }

        /**
        * @brief Get the 3D reals dimensions of all the levels of resolution of the mipmap pyramid
        *
        * @return A vector with all the levels of resolution 3D reals dimensions
        */
        std::vector<tdns::math::Vector3ui>& get_real_levels() { return _realLevels; }
        const std::vector<tdns::math::Vector3ui>& get_real_levels() const { return _realLevels; }
        
        /**
        * @brief Get the number of levels of resolution in the mipmap pyramid
        */
        size_t nb_levels() const;
        
        /**
        * @brief Get the vector containing the number of bricks per dimension, per level of resolution
        */
        std::vector<tdns::math::Vector3ui>& get_nb_bricks() { return _nbBricks; }
        const std::vector<tdns::math::Vector3ui>& get_nb_bricks() const { return _nbBricks; }

                /**
        * @brief Get the vector containing the number of bricks per dimension, per level of resolution
        */
        std::vector<tdns::math::Vector3ui>& get_nb_big_bricks() { return _nbBigBricks; }
        const std::vector<tdns::math::Vector3ui>& get_nb_big_bricks() const { return _nbBigBricks; }

        /**
        * @brief Get the vector containing the IDs of the empty bricks
        */
        std::vector<Bkey>& get_empty_bricks() { return _emptyBricks; }
        const std::vector<Bkey>& get_empty_bricks() const { return _emptyBricks; }

        /**
        * @brief Get the vector containing the histogram of the volume
        */
        std::vector<float>& get_histo() { return _histo; }
        const std::vector<float>& get_histo() const { return _histo; }

        /**
        * @brief Write the XML file descriptor with the bricks meta-data
        */
        void write_bricks_xml();

        /**
        * @brief Load the XML file descriptor with the bricks meta-data
        *
        * @return True if the loading worked
        */
        bool load();

    protected:

        std::vector<tdns::math::Vector3ui>  _initialLevels;
        std::vector<tdns::math::Vector3ui>  _realLevels;
        std::vector<tdns::math::Vector3ui>  _nbBricks;
        std::vector<tdns::math::Vector3ui>  _nbBigBricks;
        std::vector<Bkey>                   _emptyBricks;
        std::vector<float>                  _histo;
    };

    //---------------------------------------------------------------------------------------------
    inline size_t tdns::data::MetaData::nb_levels() const { return _initialLevels.size(); }
    
} //namespace data
} //namespace tdns
