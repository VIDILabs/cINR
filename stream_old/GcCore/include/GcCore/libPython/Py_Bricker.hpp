#pragma once

#include <string>
#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>

extern "C"
{
    /**
    * @brief Configuration used to brick a volume.
    *
    * @warning All fields are mandatory !
    */
    struct Py_BrickingConfiguration
    {
        const char* volumeDirectory;        ///< Folder path where the volume is.
        const char* volumeFileName;         ///< Volume name with extension.
        const char* outputDirectory;        ///< Folder path where the bricks folder will be created.
        uint32_t    level;                  ///< The level to brick. 0 for the volume given in the field "volumeFileName".
        uint32_t    startX;                 ///< The voxel index in the volume on X-axis to begin the bricking.
        uint32_t    startY;                 ///< The voxel index in the volume on Y-axis to begin the bricking.
        uint32_t    startZ;                 ///< The voxel index in the volume on Z-axis to begin the bricking.
        uint32_t    endX;                   ///< The voxel index in the volume on X-axis to end the bricking.
        uint32_t    endY;                   ///< The voxel index in the volume on Y-axis to end the bricking.
        uint32_t    endZ;                   ///< The voxel index in the volume on Z-axis to end the bricking.
        uint32_t    levelDimensionX;        ///< Volume size on X-axis for the given level.
        uint32_t    levelDimensionY;        ///< Volume size on Y-axis for the given level.
        uint32_t    levelDimensionZ;        ///< Volume size on Z-axis for the given level.
        uint32_t    brickSizeX;             ///< Size of a brick on X-axis.
        uint32_t    brickSizeY;             ///< Size of a brick on Y-axis.
        uint32_t    brickSizeZ;             ///< Size of a brick on Z-axis.
        uint32_t    bigBrickSizeX;          ///< Number of bricks in a big bricks on X-axis.
        uint32_t    bigBrickSizeY;          ///< Number of bricks in a big bricks on Y-axis.
        uint32_t    bigBrickSizeZ;          ///< Number of bricks in a big bricks on Z-axis.
        uint32_t    coveringX;              ///< Number of overlapping voxels on X-axis.
        uint32_t    coveringY;              ///< Number of overlapping voxels on Y-axis.
        uint32_t    coveringZ;              ///< Number of overlapping voxels on Z-axisSSSS.
        uint32_t    encodedBytes;            ///< Number of bytes a voxel is encoded.
        bool        compression;            ///< Save the bricks in compressed format or not.
    };

    /**
    * @brief 
    *
    * @param[in]    conf    The configuration use to brick the given level.
    */
    bool TDNS_API process_bricking(const Py_BrickingConfiguration& conf);
}

/**
* @struct Py_BrickingConfiguration
* 
* The bricks will be stored in the folder "bricks_brickSizeX_brickSizeY_brickSizeZ" in the volume directory.
* Inside this folder there are folders called "Ln" where \b n is the level set. In a "Ln" folder the brick's name is
* normalized and work like this:   
* * Ln_X_Y_Z where:
*   + Ln referes to the level of resolution.
*   + X_Y_Z referes to the position of the bricks in the volume. e.g. 0_0_0 is the first brick while 1_0_0 is the second one.
*
* @warning The values endX, endY and endZ cannot be greater than the corresponding values
* levelDimensionX, levelDimensionY and levelDimensionZ.
*
* How to cut a volume ? Let's explain it in one dimension (The behaviour is the same on all 3 dimensions (x, y, z).
* Suppose we have a volume doing with a dimension of \b 129 voxels on x-axis and we want to create bricks sized of \b 32.\n
* * \b One \b thread \b bricking   
*
* The configuration will be like this:
*
* @code
* tdns::preprocessor::Py_BrickingConfiguration conf;
* conf.startX = 0;
* conf.endX = 129;
* conf.levelDimensionX = 129;
* conf.brickSizeX = 32;
* //do not forget to fill all other fields.
* @endcode
*
* It will create 5 bricks: | [0 - 31] - [32 - 63] - [63 - 95] - [96 - 127] - [128 - 159] |
* @note Because all bricks must have the same size. The ones in the end of the volume will be filled with black voxels.
*
* * \b Multiple \b threads \b bricking   
*
* Suppose now you want to use 3 threads where one thread creates 2 bricks. The different usages will be:
* @code
* tdns::preprocessor::Py_BrickingConfiguration conf;
* // first thread
* conf.startX = 0;
* conf.endX = 64; //< upper limit included
* conf.levelDimensionX = 129;
* conf.brickSizeX = 32;
* tdns::preprocessor::process_bricking(conf):
* // second thread
* conf.startX = 64;
* conf.endX = 128; //< upper limit included
* tdns::preprocessor::process_bricking(conf):
* // third thread
* conf.startX = 128;
* conf.endX = 129; //< upper limit included
* tdns::preprocessor::process_bricking(conf):
* @endcode
* 
* Given the 3 threads, the bricks will be: | [0 - 31] - [32 - 63] | [63 - 95] - [96 - 127] | [128 - 159] |
*
* * \b Covering   
*
* The value covering when you want N overlapping voxels around a brick. Thus if the covering = 1, the bricks will have
* 30 voxels of "real" data and 1 voxel around them. It means the dimension of the real data stored in a bricks are: \n
* data = brickSize - 2 * covering \n
*
* Using this, the configuration will be:
* @code
* tdns::preprocessor::Py_BrickingConfiguration conf;
* conf.startX = 0;
* conf.endX = 129;
* conf.levelDimensionX = 129;
* conf.brickSizeX = 32;
* conf.covering = 1;
* //do not forget to fill all other fields.
* @endcode
*
* And the generated bricks (sized of 32 !!) will be: | [0 - 29] - [30 - 59] - [60 - 89] - [90 - 119] - [120 - 149] |
* Thus the first brick contains the values from -1 to 30, the second from 29 to 60 and so on.
* \note The bricks at the frontier of the volume with the overlapping not null will be filled with black and \b not
* mirror values.
* 
* * \b Big \b Bricks   
*
* The big bricks are here in order to minimize the number of files created. The values set give the number of bricks
* stored inside a big one. Thus, if bigBrickSizeX = 2, one big brick will contain 2 bricks. The configuration will be:
*
* @code
* tdns::preprocessor::Py_BrickingConfiguration conf;
* conf.startX = 0;
* conf.endX = 129;
* conf.levelDimensionX = 129;
* conf.brickSizeX = 32;
* conf.covering = 0;
* conf.bigBrickSizeX = 2;
* //do not forget to fill all other fields.
* @endcode
*
* The bricks created will be: | [0 - 31) - (32 - 63] - [63 - 95) - (96 - 127] - [128 - 159) - (160 - 191] |
* @note As said before the bricks must have the same size. So, in this example one fully black brick is added in the end.
*
* * \b Compression
* 
* If the compression is set to true all bricks will be stored as compressed bricks. The algorithm used is the LZ4 algorithm.
* Otherwise the bricks are stored as raw data.
*
* * \b Warning !!
*
* @warning The values start / end must defined in a way to be a multiple of the brickSize considering the number of bricks in a big
* brick and the covering value. Thus, setting startX = 0 and endX = 16 for one thread then startX = 16 and endX = 32 for another
* with a brick size of 32 will induce an unexpected behaviour.
*/
