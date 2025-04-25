#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libData/MetaData.hpp>
#include <GcCore/libData/VolumeConfiguration.hpp>

namespace tdns
{
namespace preprocessor
{
    struct TDNS_API MipmappingConfiguration
    {
        std::string volumeDirectory;        ///< Folder path where the volume is.
        std::string volumeFileName;         ///< Volume name with extension.
        std::string outputDirectory;        ///< Folder path where the brick files will be saved.
        tdns::math::Vector3ui   levelDimension;        ///< Volume size on X-axis for the given level.
        tdns::math::Vector3ui   downScaleRatio;
        tdns::math::Vector3ui   brickSize;    ///< Size of a brick on all axes.
        uint32_t    encodedBytes;
        uint32_t    numberChannels;
        tdns::data::DataTypeEnum    dataType; ///< Newly added type identifier
    };

    // void TDNS_API process_mipmapping(const MipmappingConfiguration& configuration);

    // void TDNS_API fill_metaData(std::vector<tdns::math::Vector3ui> &initialLevels, uint32_t levels, tdns::math::Vector3ui dimension, tdns::math::Vector3ui ratioDownScale);

    /**
    * @brief
    */
    class TDNS_API Mipmapper
    {
    public:

        /**
        * @brief Default constructor.
        */
        Mipmapper();

        /**
        * @brief Destructor.
        */
        ~Mipmapper();

        /**
         * Create the mipmap pyramid
         */
        void process(tdns::data::MetaData &metaData);
    protected:

        /**
         * @brief Initialize the mipmapper with the configuration.
         * 
         * @param  dimension            the 3D dimensions of the (original) volume
         * @param  ratioDownScale       the ratio of down sampling for each axes 
         * @param  numberEncodedBytes   the number of bytes by pixel
         * @param  numberChannels       the number of channel by pixel
         * @param  mipmapDirectory      the path of the mipmap directory
         * 
         * @return 
         */
        bool init(  tdns::math::Vector3ui &dimension,
                    tdns::math::Vector3ui &ratioDownScale,
                    tdns::data::DataTypeEnum &dataType,
                    uint32_t &numberEncodedBytes,
                    uint32_t &numberChannels,
                    std::string &mipmapDirectory);

        /**
         * @brief Call the Cuda Kernel to compute the down sampling 
         * 
         * @param output                the output buffer fill by the cuda kernel
         * @param input                 the input buffer give to the cuda kernel
         * @param resX                  the X dimension of the input
         * @param resY                  the Y dimension of the input
         * @param numberEncodedBytes    the number of bytes by pixel
         * @param numberChannels        the number of channel by pixel
         * @param ratioDownScale        the ratio of down sampling for each axes 
         */
        void down_scale(uint8_t *output,
                        const uint8_t *input,
                        const uint32_t resX,
                        const uint32_t resY,
                        const uint32_t numberEncodedBytes,
                        const uint32_t numberChannels,
                        tdns::data::DataTypeEnum dataType,
                        tdns::math::Vector3ui ratioDownScale);

        /**
         * @brief Fill the metaData object with the size of each level of the mipmap pyramid
         * 
         * @param initialLevels     the vector containing the size of each levels of the mipmap pyramid
         * @param levels            the number of level in the mipmap pyramid
         * @param dimension         the 3D dimensions of the (original) volume
         * @param ratioDownScale    the ratio of down sampling for each axes 
         */
        void fill_metaData(std::vector<tdns::math::Vector3ui> &initialLevels, uint32_t levels, tdns::math::Vector3ui dimension, tdns::math::Vector3ui ratioDownScale);

    protected:

        /**
        * Member data
        */

        uint32_t                _brickEdgeSize;         ///< Edge size of a brick.
    };

} // namespace preprocessor
} // namespace tdns