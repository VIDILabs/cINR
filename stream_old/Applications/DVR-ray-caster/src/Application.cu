#include "Application.hpp"
#include "VolumeRayCaster.hpp"

#include <vector>
#include <thread>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/VolumeConfiguration.hpp>
#include <GcCore/cuda/libPreprocessor/Mipmapper.hpp>
#include <GcCore/cuda/libPreprocessor/BrickProcessor.hpp>
#include <GcCore/cuda/libPreprocessor/BrickProcessorPredicate.hpp>
#include <GcCore/libPreprocessor/Bricker_v2.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/cuda/libGPUCache/CacheManager.hpp>

namespace tdns
{
namespace app
{
    //---------------------------------------------------------------------------------------------
    bool Application::init() const
    {
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();

        // Load configuration file
        conf.load<tdns::data::TDNSConfigurationParser>(config);

        if (!data_folder_check())
            return false;

        //step doing when opening a new volume
        std::string fileName, workingDirectory;
        conf.get_field("VolumeFile", fileName);
        conf.get_field("WorkingDirectory", workingDirectory);
        std::string volumeDirectory = workingDirectory + tdns::common::get_file_base_name(fileName) + "/";
        conf.add_field("VolumeDirectory", volumeDirectory);

        // Get the data type of the volume to visualize
        if (!conf.get_field("DataType", dataType)) {
            LOGERROR(20, "Unable to get the DataType from configuration file.");
            throw std::runtime_error("Unable to get the DataType from configuration file.");
        }

        return true;
    }

    //---------------------------------------------------------------------------------------------
    void Application::run()
    {
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        std::string volumeDirectory;
        conf.get_field("VolumeDirectory", volumeDirectory);

        int32_t gpuID;
        CUDA_SAFE_CALL(cudaGetDevice(&gpuID));

        tdns::data::MetaData volumeData;

        // Get the brick size in the configuration
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);

        // Create or load the multi-resolution bricked representation of the volume to visualize
        std::string bricksDirectory = volumeDirectory + 
            tdns::data::BricksManager::get_brick_folder(tdns::math::Vector3ui(brickSize));
        if (!tdns::common::is_dir(bricksDirectory))
        {
            LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, 
                "Bricks folder does not exist, start preprocessing... [" << bricksDirectory << "].");
            pre_process(volumeData);
        }
        else if (!volumeData.load()) return;

        LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Bricks folder found [" << bricksDirectory << "].");

        // Determine the configuration file of the volume(s) to visualize
        std::vector<tdns::data::VolumeConfiguration> volumeConfigurations(1); // only one volume here
        volumeConfigurations[0] = tdns::data::load_volume_configuration(config);

        // Cache configuration (size) 
        // (here we use only one level of pagination)
        std::vector<tdns::math::Vector3ui> blockSize(1, brickSize);
        std::vector<tdns::math::Vector3ui> cacheSize(1, tdns::math::Vector3ui(25, 25, 25));
        // cache size of (25x25x25) bricks of (16x16x16) voxels of uchar1 bytes = 25x25x25x16x16x16x1 = 61Mo
        // (Must be adapted to bricks size, voxels encoding type AND GPU available memory!).
        // (If it's too large, it will cause an [out of memory] error. If it's too small, the cache will fill up quickly and performance will suffer!)

        // Create the GPU Cache Manager
        tdns::data::CacheConfiguration cacheConfiguration;
        cacheConfiguration.CacheSize = cacheSize;
        cacheConfiguration.BlockSize = blockSize;
        cacheConfiguration.DataCacheFlags = 1;  // normalized access or not in GPU texture memory

        //===================================================================================================
        // Exemple application to use GcCore : GPU DVR ray-caster
        //===================================================================================================

        auto dtype = tdns::data::getDataTypeEnum(dataType);

#define CALL_APP(TYPE)                                                          \
        {                                                                       \
            auto* cacheManager = new tdns::gpucache::CacheManager<TYPE>         \
                (volumeConfigurations[0], cacheConfiguration, gpuID);           \
            tdns::graphics::display_volume_raycaster(cacheManager, volumeData); \
            delete cacheManager;                                                \
        }

        if      (dtype == tdns::data::TYPE_UCHAR)  CALL_APP(uchar1)
        else if (dtype == tdns::data::TYPE_USHORT) CALL_APP(ushort1)
        else if (dtype == tdns::data::TYPE_UINT)   CALL_APP(uint1)
        else if (dtype == tdns::data::TYPE_CHAR)   CALL_APP(char1)
        else if (dtype == tdns::data::TYPE_SHORT)  CALL_APP(short1)
        else if (dtype == tdns::data::TYPE_INT)    CALL_APP(int1)
        else if (dtype == tdns::data::TYPE_FLOAT)  CALL_APP(float1)
        // else if (dtype == tdns::data::TYPE_DOUBLE) CALL_APP(double1)
        else throw std::runtime_error("Unsupported data type: " + std::to_string(dtype));

        // CUDA_SYNC_CHECK();
    }

    //---------------------------------------------------------------------------------------------
    bool Application::data_folder_check() const
    {
        if (tdns::common::is_dir("data")) return true;
        
        if (tdns::common::is_file("data"))
        {
            LOGFATAL(10, "Data already exist next to the binary and is not a folder.");
            return false;
        }

        LOGTRACE(10, tdns::common::log_details::Verbosity::INSANE, "Create the folder \"data\" next to the binary.");
        tdns::common::create_folder("data");
        return true;
    }

    //---------------------------------------------------------------------------------------------
    template<typename DataType>
    void pre_process_empty_and_histogram(tdns::data::MetaData &volumeData, double threshold)
    {
        tdns::preprocessor::BrickProcessor<DataType> brickProcessor(volumeData);
        // // Process volume histogram
        // brickProcessor.process_histo();
        // Process empty bricks
        void *d_threshold; 
        DataType h_threshold;
        h_threshold.x = threshold;
        CUDA_SAFE_CALL(cudaMalloc(&d_threshold, sizeof(h_threshold.x)));
        CUDA_SAFE_CALL(cudaMemcpy(d_threshold, &h_threshold.x, sizeof(h_threshold.x), cudaMemcpyHostToDevice));
        brickProcessor.template process_empty<tdns::preprocessor::BrickProcessorGenericPredicate_1<DataType>>(d_threshold);
        CUDA_SAFE_CALL(cudaFree(d_threshold));
    }

    void Application::pre_process(tdns::data::MetaData &volumeData) const
    {
        std::cout << "Start pre-processing (see log file) ..." << std::endl;

        // Mipmapping
        tdns::preprocessor::Mipmapper mipmapper;
        mipmapper.process(volumeData);

        // Bricking
        std::vector<tdns::math::Vector3ui> levels = volumeData.get_initial_levels();
        pre_process_bricking(volumeData, levels);

        // PROCESS EMPTY BRICKS AND VOLUME HISTOGRAM
        auto dtype = tdns::data::getDataTypeEnum(dataType);
        double threshold = 0.0; // TODO: get from config file
        if      (dtype == tdns::data::TYPE_UCHAR)  pre_process_empty_and_histogram<uchar1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_USHORT) pre_process_empty_and_histogram<ushort1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_UINT)   pre_process_empty_and_histogram<uint1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_CHAR)   pre_process_empty_and_histogram<char1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_SHORT)  pre_process_empty_and_histogram<short1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_INT)    pre_process_empty_and_histogram<int1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_FLOAT)  pre_process_empty_and_histogram<float1>(volumeData, threshold);
        else if (dtype == tdns::data::TYPE_DOUBLE) pre_process_empty_and_histogram<double1>(volumeData, threshold);
        else throw std::runtime_error("Unsupported data type: " + std::to_string(dtype));
        volumeData.write_bricks_xml();
    }

    //---------------------------------------------------------------------------------------------
    void Application::pre_process_bricking(tdns::data::MetaData &volumeData, const std::vector<tdns::math::Vector3ui> &levels) const
    {
        tdns::preprocessor::BrickingConfiguration configuration;
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        //brick Size
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);
        configuration.brickSize = tdns::math::Vector3ui(brickSize);
        //EncodedByte
        conf.get_field("NumberEncodedBytes", configuration.encodedBytes);
        //covering
        conf.get_field("VoxelCovering", configuration.covering);
        //Volumedirectory
        conf.get_field("VolumeDirectory", configuration.volumeDirectory);
        //volume file name
        conf.get_field("VolumeFile", configuration.volumeFileName);
        //Outputdirectory
        conf.get_field("VolumeDirectory", configuration.outputDirectory);
        //compression ?
        configuration.compression = true;
        //big brick size
        tdns::math::Vector3ui bigBrickSize;
        conf.get_field("BigBrickSizeX", bigBrickSize[0]);
        conf.get_field("BigBrickSizeY", bigBrickSize[1]);
        conf.get_field("BigBrickSizeZ", bigBrickSize[2]);
        configuration.bigBrickSize = bigBrickSize;

        //fill volumeData.
        tdns::preprocessor::init_meta_data(volumeData, configuration, levels);

        std::vector<std::thread> threads(levels.size());
        for (uint32_t i = 0; i < threads.size(); ++i)
        {
            threads[i] = std::thread([&, i, configuration]() mutable
            {
                configuration.level = i;
                configuration.levelDimensionX = levels[i][0];
                configuration.levelDimensionY = levels[i][1];
                configuration.levelDimensionZ = levels[i][2];

                configuration.startX = configuration.startY = configuration.startZ = 0;
                configuration.endX = configuration.levelDimensionX = levels[i][0];
                configuration.endY = configuration.levelDimensionY = levels[i][1];
                configuration.endZ = configuration.levelDimensionZ = levels[i][2];

                tdns::preprocessor::process_bricking(configuration);
            });
        }

        for (size_t i = 0; i < threads.size(); ++i)
            if (threads[i].joinable())
                threads[i].join();
    }
} // namespace app
} // namespace tdns