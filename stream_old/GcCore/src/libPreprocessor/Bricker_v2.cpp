#include <GcCore/libPreprocessor/Bricker_v2.hpp>

#include <memory>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>

#include <GcCore/libData/FilesManager.hpp>
#include <GcCore/libData/Brick.hpp>
#include <GcCore/libData/BricksManager.hpp>

namespace tdns
{
namespace preprocessor
{
    /**
    * @brief Give the file name of the volume regarding its level.
    * @param The configuration used.
    * @return File of the volume.
    */
    std::string get_file_path(const BrickingConfiguration& conf);

    /**
    * @brief Check if the "Bricks" folder exist, if not create it and check if the
    * level folder "LX" exists in it, if not create it.
    * @param The configuration used.
    */
    void check_level_directory(const BrickingConfiguration& conf);

    /**
    * @brief Transform a 3D position into a 1D position.
    *
    * @param[in]    conf    The configuration to know the volume size.
    * @param[in]    x       The x positon.
    * @param[in]    y       The y positon.
    * @param[in]    z       The z positon.
    *
    * @return The 1D position.
    */
    uint64_t get_linear_position(const BrickingConfiguration &conf, uint32_t x, uint32_t y, uint32_t z);

    std::vector<uint8_t>::iterator get_iterator_from_indexes(const BrickingConfiguration &conf,
        std::vector<uint8_t> &data, uint64_t x, uint64_t y, uint64_t z);

    //---------------------------------------------------------------------------------------------------
    void init_meta_data(tdns::data::MetaData &metaData, const BrickingConfiguration& conf,
        const std::vector<tdns::math::Vector3ui> &levels)
    {
        uint32_t brickSizeWithoutCoveringX = conf.brickSize[0] - 2 * conf.covering;
        uint32_t brickSizeWithoutCoveringY = conf.brickSize[1] - 2 * conf.covering;
        uint32_t brickSizeWithoutCoveringZ = conf.brickSize[2] - 2 * conf.covering;
        std::vector<tdns::math::Vector3ui> &realLevelsSize = metaData.get_real_levels();
        std::vector<tdns::math::Vector3ui> &nbBricks = metaData.get_nb_bricks();
        std::vector<tdns::math::Vector3ui> &nbBigBricks = metaData.get_nb_big_bricks();
        for (uint32_t i = 0; i < levels.size(); ++i)
        {
            uint32_t nbBricksX = static_cast<uint32_t>(std::ceil((float)levels[i][0] / (float)brickSizeWithoutCoveringX));
            uint32_t nbBricksY = static_cast<uint32_t>(std::ceil((float)levels[i][1] / (float)brickSizeWithoutCoveringY));
            uint32_t nbBricksZ = static_cast<uint32_t>(std::ceil((float)levels[i][2] / (float)brickSizeWithoutCoveringZ));

            nbBricksX = static_cast<uint32_t>(std::ceil((float)nbBricksX / (float)conf.bigBrickSize[0])) * conf.bigBrickSize[0];
            nbBricksY = static_cast<uint32_t>(std::ceil((float)nbBricksY / (float)conf.bigBrickSize[1])) * conf.bigBrickSize[1];
            nbBricksZ = static_cast<uint32_t>(std::ceil((float)nbBricksZ / (float)conf.bigBrickSize[2])) * conf.bigBrickSize[2];
            nbBricks.push_back(tdns::math::Vector3ui(nbBricksX, nbBricksY, nbBricksZ));

            uint32_t nbBigBricksX = nbBricksX / conf.bigBrickSize[0];
            uint32_t nbBigBricksY = nbBricksY / conf.bigBrickSize[1];
            uint32_t nbBigBricksZ = nbBricksZ / conf.bigBrickSize[2];
            nbBigBricks.push_back(tdns::math::Vector3ui(nbBigBricksX, nbBigBricksY, nbBigBricksZ));
            

            uint32_t realSizeX = nbBricksX * brickSizeWithoutCoveringX;
            uint32_t realSizeY = nbBricksY * brickSizeWithoutCoveringY;
            uint32_t realSizeZ = nbBricksZ * brickSizeWithoutCoveringZ;
            realLevelsSize.push_back(tdns::math::Vector3ui(realSizeX, realSizeY, realSizeZ));
        }
    }

    //---------------------------------------------------------------------------------------------------
    bool process_bricking(const BrickingConfiguration& conf)
    {
        std::string filePath = get_file_path(conf);

        LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "Starting bricking volume [" << filePath << "] - level " << conf.level << ".");
        std::unique_ptr<tdns::data::AbstractFile> file = tdns::data::FilesManager::get_instance().get_file_from_path(filePath);

        //check the file
        if (!file)
        {
            LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "File [" << filePath << "] does not exist.");
            return false;
        }
        //open it
        file->open();

        //Preliminary check
        check_level_directory(conf);

        //avoid multiple read
        uint32_t covering = conf.covering;
        uint32_t brickEdgeSizeX = conf.brickSize[0];
        uint32_t brickEdgeSizeY = conf.brickSize[1];
        uint32_t brickEdgeSizeZ = conf.brickSize[2];
        uint32_t numberEncodedBytes = conf.encodedBytes;
        tdns::math::Vector3ui brickSize_noCovering(
            brickEdgeSizeX - 2 * covering,
            brickEdgeSizeY - 2 * covering,
            brickEdgeSizeZ - 2 * covering
        );

        //used for logs
        float maxX = static_cast<float>(conf.endX - conf.startX) / brickSize_noCovering[0];
        float maxY = static_cast<float>(conf.endY - conf.startY) / brickSize_noCovering[1];
        float maxZ = static_cast<float>(conf.endZ - conf.startZ) / brickSize_noCovering[2];
        uint64_t max = static_cast<uint64_t>(std::ceil(maxX / conf.bigBrickSize[0]) *
            std::ceil(maxY / conf.bigBrickSize[1]) *
            std::ceil(maxZ / conf.bigBrickSize[2]));
        uint64_t current = 1;

        //absolute position of the brick in the volume (0 to volume size)
        tdns::math::Vector3ui absBrick(conf.startX, conf.startY, conf.startZ);
        //absolute position of the big brick in the volume (0 to volume size)
        tdns::math::Vector3ui absBigBrick(conf.startX, conf.startY, conf.startZ);
        //absolute position in the volume (0 to volume size)
        int32_t absX = conf.startX, absY = conf.startY, absZ = conf.startZ;
        //3D brick indexes (from 0 to max number of bigbricks [on all axes])
        tdns::math::Vector3ui initialBigBrickPosition;
        initialBigBrickPosition[0] = (uint32_t)std::ceil((float)conf.startX / brickSize_noCovering[0]);
        initialBigBrickPosition[1] = (uint32_t)std::ceil((float)conf.startY / brickSize_noCovering[1]);
        initialBigBrickPosition[2] = (uint32_t)std::ceil((float)conf.startZ / brickSize_noCovering[2]);
        initialBigBrickPosition /= conf.bigBrickSize;
        //brick indexes in the big brick (from 0 to max numver of bricks in big bricks [on all axes])
        tdns::math::Vector3ui initialBrickPosition(0);
        tdns::math::Vector3ui brickPosition(initialBrickPosition);
        //3D brick indexes (from 0 to max number of bricks [on all axes])
        tdns::math::Vector3ui bigBrickPosition(initialBigBrickPosition);

        tdns::data::Brick brick(0, conf.bigBrickSize * conf.brickSize, numberEncodedBytes);
        while (true)
        {
            //create the brick
            brick.fill(0); //reset the values
            brick.set_position(bigBrickPosition);
            brick.set_level(conf.level);

            brickPosition = tdns::math::Vector3ui(0);
            while (true)
            {
                //set the absolute position of the brick given the big brick position and the brick position inside the big one.
                absBrick = (bigBrickPosition * conf.bigBrickSize + brickPosition) * brickSize_noCovering;

                //position in the big brick the given small brick
                tdns::math::Vector3ui absPositionInBigBrick = brickPosition * conf.brickSize;

                uint32_t brickX = 0, brickY = 0, brickZ = 0;
                //check the border on the x axis
                if (absBrick[0] + brickX == 0) //if we are in the first columns
                    brickX += covering;
                absX = absBrick[0] + brickX - covering;

                //for each slice of the brick
                for (brickZ = 0; brickZ < brickEdgeSizeZ; ++brickZ)
                {
                    absZ = absBrick[2] + brickZ - covering;
                    if (absZ < 0 && brickZ < covering) continue; //the covering slices of the volume
                    if (absZ >= static_cast<int32_t>(conf.levelDimensionZ)) break; //the last slides of the volume

                    //for each row of the brick
                    for (brickY = 0; brickY < brickEdgeSizeY; ++brickY)
                    {
                        absY = absBrick[1] + brickY - covering;
                        if (absY < 0 && brickY < covering) continue; //the covering rows of the volume
                        if (absY >= static_cast<int32_t>(conf.levelDimensionY)) break; //the last rows of the volume

                        //set Cursor
                        file->set_absolute_cursor_position(get_linear_position(
                            conf,
                            absX,
                            absY,
                            absZ));

                        //calculate the length to read
                        uint32_t XToRead;
                        if (absX + brickEdgeSizeX > conf.levelDimensionX)
                        {
                            if (absX == 0) //case were the dimension on X axis is < to the brick size
                                XToRead = conf.levelDimensionX;
                            else
                                XToRead = conf.levelDimensionX - absX;
                        }
                        else
                            XToRead = brickEdgeSizeX - brickX; //size of the brick - where we are in the brick on X axis

                        //read from file
                        auto itBegin = get_iterator_from_indexes(conf, brick.get_data(),
                            absPositionInBigBrick[0] + brickX,
                            absPositionInBigBrick[1] + brickY,
                            absPositionInBigBrick[2] + brickZ);
                        file->read(&(*itBegin), XToRead * numberEncodedBytes);
                    }
                }

                //change the position of the small brick
                if ((bigBrickPosition[0] * conf.bigBrickSize[0] + brickPosition[0] + 1) * brickSize_noCovering[0] < conf.endX
                    && (brickPosition[0] + 1) < conf.bigBrickSize[0])
                    ++brickPosition[0];
                else
                {
                    brickPosition[0] = initialBrickPosition[0];
                    if ((bigBrickPosition[1] * conf.bigBrickSize[1] + brickPosition[1] + 1) * brickSize_noCovering[1] < conf.endY
                        && (brickPosition[1] + 1) < conf.bigBrickSize[1])
                        ++brickPosition[1];
                    else
                    {
                        brickPosition[1] = initialBrickPosition[1];
                        if ((bigBrickPosition[2] * conf.bigBrickSize[2] + brickPosition[2] + 1) * brickSize_noCovering[2] < conf.endZ
                            && (brickPosition[2] + 1) < conf.bigBrickSize[2])
                            ++brickPosition[2];
                        else
                            break;
                    }
                }
            }

            //save brick
            float compression = tdns::data::BricksManager::write_brick(conf.outputDirectory, brick, conf.brickSize, conf.compression);

            LOGTRACE(20, tdns::common::log_details::Verbosity::INSANE, "Bricking level "
                << conf.level << " - " << current++ << " / " << max << " - " << compression << "% compression.");

            //change position of the big brick
            if ((bigBrickPosition[0] + 1) * conf.bigBrickSize[0]  * brickSize_noCovering[0] < conf.endX)
                ++bigBrickPosition[0];
            else
            {
                bigBrickPosition[0] = initialBigBrickPosition[0];
                if ((bigBrickPosition[1] + 1) * conf.bigBrickSize[1] * brickSize_noCovering[1] < conf.endY)
                    ++bigBrickPosition[1];
                else
                {
                    bigBrickPosition[1] = initialBigBrickPosition[1];
                    if ((bigBrickPosition[2] + 1) * conf.bigBrickSize[2] * brickSize_noCovering[2] < conf.endZ)
                        ++bigBrickPosition[2];
                    else
                        break;
                }
            }
        }

        LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "End bricking volume [" << filePath << "] - level " << conf.level << ".");

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
    std::string get_file_path(const BrickingConfiguration& conf)
    {
        std::string filePath;

        //particular case, it's the full resolution volume
        if (conf.level == 0)
        {
            filePath = conf.volumeDirectory + conf.volumeFileName;
        }
        else
        {
            filePath = conf.volumeDirectory + "mipmap/L" + std::to_string(conf.level) + ".raw";
        }
        return filePath;
    }

    //---------------------------------------------------------------------------------------------------
    void check_level_directory(const BrickingConfiguration& conf)
    {
        tdns::data::BricksManager::check_level_directory(
            conf.volumeDirectory,
            conf.level,
            conf.brickSize);
    }

    //---------------------------------------------------------------------------------------------------
    uint64_t get_linear_position(const BrickingConfiguration &conf, uint32_t x, uint32_t y, uint32_t z)
    {
        uint64_t X = static_cast<uint64_t>(x);
        uint64_t Y = static_cast<uint64_t>(y);
        uint64_t Z = static_cast<uint64_t>(z);
        uint64_t dX = static_cast<uint64_t>(conf.levelDimensionX);
        uint64_t dY = static_cast<uint64_t>(conf.levelDimensionY);
        uint64_t E = static_cast<uint64_t>(conf.encodedBytes);
        return (X + dX * (Y + Z * dY)) * E;
        // return (X + Y * dX + Z * dX * dY) * E;
    }

    //---------------------------------------------------------------------------------------------------
    std::vector<uint8_t>::iterator get_iterator_from_indexes(const BrickingConfiguration &conf,
        std::vector<uint8_t> &data, uint64_t x, uint64_t y, uint64_t z)
    {
        return data.begin() + (x + conf.brickSize[0] * conf.bigBrickSize[0] * (y + z * conf.brickSize[1] * conf.bigBrickSize[1])) * conf.encodedBytes;
    }
} // namespace preprocessor
} // namespace tdns