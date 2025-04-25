#include "cachemanager.h"

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/VolumeConfiguration.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/LogCategory.hpp>
#include <GcCore/libCommon/Logger/LoggerFormatterFileStd.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libPreprocessor/Bricker_v2.hpp>
#include <GcCore/cuda/libPreprocessor/Mipmapper.hpp>
#include <GcCore/cuda/libPreprocessor/BrickProcessor.hpp>
#include <GcCore/cuda/libPreprocessor/BrickProcessorPredicate.hpp>

// NOTE: typically the best practice is to include standard library in the end, unless it causes issues.
#include <string>
#include <thread>
#include <memory>

namespace tdns::app::LogCategories {
  const static tdns::common::LogCategory cat_default(0, "Default", "Default category to dump logs.", tdns::common::log_details::Verbosity::INSANE);
  const static tdns::common::LogCategory cat_system(10, "System", "All logs related to the system. File access, etc.", tdns::common::log_details::Verbosity::INSANE);
  const static tdns::common::LogCategory cat_Preprocessor(20, "Preprocessor", "All logs related to the preprocessing of the volumes", tdns::common::log_details::Verbosity::INSANE);
  const static tdns::common::LogCategory cat_Graphics(30, "Graphics", "All logs related to the graphics part. SDL, window, etc.", tdns::common::log_details::Verbosity::INSANE);
  const static tdns::common::LogCategory cat_GPUCache(40, "GPUCache", "All logs related to the cache system.", tdns::common::log_details::Verbosity::INSANE);
  void load_categories_in_logger() {
      tdns::common::Logger::get_instance().add_category(cat_default);
      tdns::common::Logger::get_instance().add_category(cat_system);
      tdns::common::Logger::get_instance().add_category(cat_Preprocessor);
      tdns::common::Logger::get_instance().add_category(cat_Graphics);
      tdns::common::Logger::get_instance().add_category(cat_GPUCache);
  }
}

namespace ovr::nncache {

tdns::common::LoggerFormatterFileStd log_formatter; // has to be global variable

// NOTE: static makes this function is local to this file, and it is invisible to other source files, so we can avoid name conflicts.

// template<typename DataType>
// static void pre_process_empty_and_histogram(tdns::data::MetaData &volumeData, double threshold) {
//   tdns::preprocessor::BrickProcessor<DataType> brickProcessor(volumeData);
//   // // Process volume histogram
//   // brickProcessor.process_histo();
//   // Process empty bricks
//   void *d_threshold; 
//   DataType h_threshold;
//   h_threshold.x = threshold;
//   CUDA_SAFE_CALL(cudaMalloc(&d_threshold, sizeof(h_threshold.x)));
//   CUDA_SAFE_CALL(cudaMemcpy(d_threshold, &h_threshold.x, sizeof(h_threshold.x), cudaMemcpyHostToDevice));
//   brickProcessor.template process_empty<tdns::preprocessor::BrickProcessorGenericPredicate_1<DataType>>(d_threshold);
//   CUDA_SAFE_CALL(cudaFree(d_threshold));
// }

// static void pre_process_bricking(tdns::data::MetaData &volumeData, const std::vector<tdns::math::Vector3ui> &levels) {
//   tdns::preprocessor::BrickingConfiguration configuration;
//   tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
//   // brick Size
//   uint32_t brickSize;
//   conf.get_field("BrickSize", brickSize);
//   configuration.brickSize = tdns::math::Vector3ui(brickSize);
//   // EncodedByte
//   conf.get_field("NumberEncodedBytes", configuration.encodedBytes);
//   // covering
//   conf.get_field("VoxelCovering", configuration.covering);
//   // Volumedirectory
//   conf.get_field("VolumeDirectory", configuration.volumeDirectory);
//   // volume file name
//   conf.get_field("VolumeFile", configuration.volumeFileName);
//   // Outputdirectory
//   conf.get_field("VolumeDirectory", configuration.outputDirectory);
//   // compression or not
//   configuration.compression = true;
//   // big brick size
//   tdns::math::Vector3ui bigBrickSize;
//   conf.get_field("BigBrickSizeX", bigBrickSize[0]);
//   conf.get_field("BigBrickSizeY", bigBrickSize[1]);
//   conf.get_field("BigBrickSizeZ", bigBrickSize[2]);
//   configuration.bigBrickSize = bigBrickSize;

//   // fill volumeData.
//   tdns::preprocessor::init_meta_data(volumeData, configuration, levels);
//   std::vector<std::thread> threads(levels.size());
//   for (uint32_t i = 0; i < threads.size(); ++i) {
//     threads[i] = std::thread([&, i, configuration]() mutable {
//       configuration.level = i;
//       configuration.levelDimensionX = levels[i][0];
//       configuration.levelDimensionY = levels[i][1];
//       configuration.levelDimensionZ = levels[i][2];
//       configuration.startX = configuration.startY = configuration.startZ = 0;
//       configuration.endX = configuration.levelDimensionX = levels[i][0];
//       configuration.endY = configuration.levelDimensionY = levels[i][1];
//       configuration.endZ = configuration.levelDimensionZ = levels[i][2];
//       tdns::preprocessor::process_bricking(configuration);
//     });
//   }

//   for (size_t i = 0; i < threads.size(); ++i)
//       if (threads[i].joinable())
//           threads[i].join();
// }

// static void pre_process(tdns::data::MetaData &volumeData, std::string dataType) {
//   std::cout << "Start pre-processing (see log file) ..." << std::endl;
  
  // // Mipmapping
  // std::cout << " -- Mipmapping ..." << std::endl;
  // tdns::preprocessor::Mipmapper mipmapper;
  // mipmapper.process(volumeData);
  
//   // Bricking
//   std::cout << " -- Bricking ..." << std::endl;
//   std::vector<tdns::math::Vector3ui> levels = volumeData.get_initial_levels();
//   pre_process_bricking(volumeData, levels);

//   // // PROCESS EMPTY BRICKS AND VOLUME HISTOGRAM
//   // auto dtype = tdns::data::getDataTypeEnum(dataType);
//   // double threshold = 0.0; // TODO: get from config file
//   // if      (dtype == tdns::data::TYPE_UCHAR)  pre_process_empty_and_histogram<uchar1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_USHORT) pre_process_empty_and_histogram<ushort1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_UINT)   pre_process_empty_and_histogram<uint1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_CHAR)   pre_process_empty_and_histogram<char1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_SHORT)  pre_process_empty_and_histogram<short1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_INT)    pre_process_empty_and_histogram<int1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_FLOAT)  pre_process_empty_and_histogram<float1>(volumeData, threshold);
//   // else if (dtype == tdns::data::TYPE_DOUBLE) pre_process_empty_and_histogram<double1>(volumeData, threshold);
//   // else throw std::runtime_error("Unsupported data type: " + std::to_string(dtype));

//   volumeData.write_bricks_xml();
// }

std::string get_cache_datapath(const std::string& configFilename) {
  tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
  conf.load<tdns::data::TDNSConfigurationParser>(configFilename);
  std::string volumeFileName;
  conf.get_field("VolumeFile", volumeFileName);
  std::string workingDirectory;
  conf.get_field("WorkingDirectory", workingDirectory);
  return workingDirectory + tdns::common::get_file_base_name(volumeFileName) + "/" + volumeFileName;
}

// This function is exposed in the device.cu file, so should not be static.
template <typename T>
static std::unique_ptr<tdns::gpucache::CacheManager<T>> 
create_cache_manager(std::string configFilename, vec3i capacity, int nbLevels, uint32_t& maxLoD) 
{
  // ---------------------------
  // main.cpp
  tdns::app::LogCategories::load_categories_in_logger();
  tdns::common::Logger::get_instance().open("./3dns", log_formatter, true);

  // ---------------------------
  // Application.cu: init()
  
  // Load configuration file
  tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
  conf.load<tdns::data::TDNSConfigurationParser>(configFilename);
  
  // Get fields from config
  std::string volumeFileName;
  conf.get_field("VolumeFile", volumeFileName);
  std::string workingDirectory;
  conf.get_field("WorkingDirectory", workingDirectory);
  std::string volumeDirectory = workingDirectory + tdns::common::get_file_base_name(volumeFileName) + "/";
  conf.add_field("VolumeDirectory", volumeDirectory);

  // Get the data type of the volume to visualize
  std::string dataType;
  if (!conf.get_field("DataType", dataType)) {
      LOGERROR(20, "Unable to get the DataType from configuration file.");
      throw std::runtime_error("Unable to get the DataType from configuration file.");
  }

  // ---------------------------
  // Application.cu: run()

  int32_t gpuID;
  CUDA_SAFE_CALL(cudaGetDevice(&gpuID));
  tdns::data::MetaData volumeData;

  uint32_t brickSize;
  conf.get_field("BrickSize", brickSize);

  // Create or load the multi-resolution bricked volume representation

  // std::string bricksDirectory = volumeDirectory + 
  //   tdns::data::BricksManager::get_brick_folder(tdns::math::Vector3ui(brickSize));
  // if (!tdns::common::is_dir(bricksDirectory)) {
  //     LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, 
  //         "Bricks folder does not exist, start preprocessing... [" << bricksDirectory << "].");
  //     pre_process(volumeData, dataType);
  // }
  // else if (!volumeData.load()) throw std::runtime_error("failed to load");
  // LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Bricks folder found [" << bricksDirectory << "].");


  // Create Volume and Cache configerations
  tdns::data::VolumeConfiguration volumeConfiguration;
  volumeConfiguration = tdns::data::load_volume_configuration(configFilename);
  std::vector<tdns::math::Vector3ui> blockSize(1, brickSize);
  std::vector<tdns::math::Vector3ui> cacheSize(1, 
    tdns::math::Vector3ui(capacity.x, capacity.y, capacity.z)
  );

  // TODO: edit volumeconfig.volumeFilename with new model id (pass in id)
  
  // cache size of (25x25x25) bricks of (16x16x16) voxels of uchar1 bytes = 25x25x25x16x16x16x1 = 61Mo
  // (Must be adapted to bricks size, voxels encoding type AND GPU available memory!).
  // (If it's too large, it will cause an [out of memory] error. If it's too small, the cache will fill up quickly and performance will suffer!)
  tdns::data::CacheConfiguration cacheConfiguration;
  cacheConfiguration.CacheSize = cacheSize;
  cacheConfiguration.BlockSize = blockSize;
  // This flag decides whether to use "get_normalized" or "get" in the kernel
  cacheConfiguration.DataCacheFlags = 1;  // normalized access or not in GPU texture memory

  // build CacheManager
  auto cacheManager = std::make_unique<tdns::gpucache::CacheManager<T>>(volumeConfiguration, cacheConfiguration, gpuID);
  
  // ---------------------------
  // VolumeRayCaster.cu: display_volume_raycaster()
  // Load the pre-computed histogram of the volume
  // std::vector<float> &histo = volumeData.get_histo();
  // size_t histoSize = histo.size();

  uint32_t x, y, z;
  conf.get_field("size_X", x);
  conf.get_field("size_Y", y);
  conf.get_field("size_Z", z);

  // const uint32_t maxDim = std::max(x, std::max(y, z));
  // const int levelMax = static_cast<uint32_t>(std::ceil(std::log(maxDim / brickSize) / std::log(2)));
  const int levelMax = volumeConfiguration.NbLevels;

  // Get LoD MetaData
  // maxLoD = volumeData.nb_levels() - 1;
  // maxLoD = std::min(nbLevels, levelMax);
  maxLoD = levelMax;

  return std::move(cacheManager); // because it is an unique_ptr, there is no copy constructor.
}

#define CREATE_CACHE_MANAGER(handle, TYPE) {                               \
  handle = create_cache_manager<TYPE>(config, capacity, nbLevels, maxLoD); \
  m->cache = handle.get(); break;                                          \
}

std::shared_ptr<OpaqueCacheManager> 
create_cache_manager(const std::string& config, const vec3i& capacity, int nbLevels, ValueType type) 
{ 
  std::shared_ptr<OpaqueCacheManager> m = std::make_shared<OpaqueCacheManager>();
  auto& maxLoD = m->max_lod;
  m->type = type;
  // switch (type) {
  // case VALUE_TYPE_UINT8:  CREATE_CACHE_MANAGER(m->m_uchar,  uchar1 )
  // case VALUE_TYPE_INT8:   CREATE_CACHE_MANAGER(m->m_char,   char1  )
  // case VALUE_TYPE_UINT16: CREATE_CACHE_MANAGER(m->m_ushort, ushort1)
  // case VALUE_TYPE_INT16:  CREATE_CACHE_MANAGER(m->m_short,  short1 )
  // case VALUE_TYPE_UINT32: CREATE_CACHE_MANAGER(m->m_uint,   uint1  )
  // case VALUE_TYPE_INT32:  CREATE_CACHE_MANAGER(m->m_int,    int1   )
  // case VALUE_TYPE_FLOAT:  CREATE_CACHE_MANAGER(m->m_float,  float1 )
  // default: throw std::runtime_error("unsupported type encountered: " + std::to_string(type));
  // }
  m->m_float = create_cache_manager<float1>(config, capacity, nbLevels, maxLoD); 
  m->cache = m->m_float.get();   
  return m;
}

}
