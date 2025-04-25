#include <GcCore/cuda/libPreprocessor/Mipmapper.hpp>

#include <GcCore/cuda/libPreprocessor/Constants.hpp>

#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <memory>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/AbstractFile.hpp>
#include <GcCore/libData/RAWFile.hpp>
#include <GcCore/libData/FilesManager.hpp>

#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Memory.hpp>

#include <GcCore/libTinyXml/tinyxml2.h>

namespace tdns
{
namespace preprocessor
{
    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
    template<typename T>
    __global__ void down_sampling(uint8_t *input,
        uint8_t *output,
        const uint32_t inputWidth,
        const uint32_t inputHeight,
        const uint32_t inputWidthStep,
        const uint32_t outputWidthStep,
        const uint32_t numberEncodedBytes,
        const uint32_t numberChannels,
        const uint3 pixelGroupSize)
    {
        //2D Index of current thread
        const uint32_t outputXIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t outputYIndex = blockIdx.y * blockDim.y + threadIdx.y;

        const uint32_t outputWidth = inputWidth / pixelGroupSize.x * numberChannels;
        const uint32_t outputHeight = inputHeight / pixelGroupSize.y;

        //Only valid threads perform memory I/O
        if ((outputXIndex >= outputWidth) || (outputYIndex >= outputHeight))
            return;

        // Starting location of current pixel in output
        uint32_t output_tid = (outputYIndex * outputWidthStep) + (outputXIndex * (numberEncodedBytes / numberChannels));

        // Compute the size of the area of pixels to be resized to a single pixel
        const uint32_t pixelGroupArea = pixelGroupSize.x * pixelGroupSize.y * pixelGroupSize.z;

        // Compute the pixel group area in the input image
        const uint32_t inputXIndexStart = ((outputXIndex / numberChannels) * pixelGroupSize.x) * numberEncodedBytes + (outputXIndex % numberChannels);
        const uint32_t inputXIndexEnd = inputXIndexStart + (pixelGroupSize.x - 1) * numberEncodedBytes;
        const uint32_t inputYIndexStart = outputYIndex * pixelGroupSize.y;
        const uint32_t inputYIndexEnd = inputYIndexStart + pixelGroupSize.y;

        if (numberChannels == 1)
        {
            double channelSum;
            T *ptrInput = reinterpret_cast<T*>(input);
            T *ptrOutput = reinterpret_cast<T*>(output);
            for (uint32_t inputZIndex = 0; inputZIndex < pixelGroupSize.z; ++inputZIndex)
            for (uint32_t inputYIndex = inputYIndexStart; inputYIndex < inputYIndexEnd; ++inputYIndex)
            for (uint32_t inputXIndex = inputXIndexStart; inputXIndex <= inputXIndexEnd; inputXIndex += numberEncodedBytes)
            {
                uint32_t input_tid = inputZIndex * (inputWidthStep * inputHeight) + (inputYIndex * inputWidthStep) + inputXIndex;
                T pixelValue = ptrInput[input_tid / numberEncodedBytes];
                channelSum += static_cast<double>(pixelValue);
            }
            T out = channelSum / static_cast<double>(pixelGroupArea);
            ptrOutput[output_tid / numberEncodedBytes] = out;
        }
        else
        {
            printf("ERROR : number of channels not supported\n");
        }
    }

    //---------------------------------------------------------------------------------------------------

    void down_scale(uint8_t *output,
        const uint8_t *input,
        const uint32_t resX,
        const uint32_t resY,
        const uint32_t numberEncodedBytes,
        const uint32_t numberChannels,
        tdns::data::DataTypeEnum dataType,
        tdns::math::Vector3ui ratioDownScale)
    {
        //device buffers
        uint8_t *d_input, *d_output;

        uint32_t inputSize = resX * resY * numberEncodedBytes * ratioDownScale[2];
        uint32_t outputSize = (resX / ratioDownScale[0]) * (resY / ratioDownScale[1]) * numberEncodedBytes;

        // Size in bytes of a "line" of the image for the input and for the output.
        uint32_t inputWidthBytes = resX * numberEncodedBytes;
        uint32_t outputWidthBytes = (resX / ratioDownScale[0]) * numberEncodedBytes;

        // Allocate device memory
        CUDA_SAFE_CALL(cudaMalloc<uint8_t>(&d_input, inputSize));
        CUDA_SAFE_CALL(cudaMalloc<uint8_t>(&d_output, outputSize));

        // Copy data from host input to device memory
        CUDA_SAFE_CALL(cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice));

        // Specify a reasonable block size (number of threads per block)
        const dim3 block(16, 16);
        // Calculate grid size to cover the whole image (number of blocks)
        const dim3 grid((resX * numberChannels / ratioDownScale[0] + block.x - 1) / block.x, (resY / ratioDownScale[1] + block.y - 1) / block.y);

#define CALL_DOWNSAMPLE_SAMPLING(TYPE)  \
  down_sampling <TYPE> <<<grid, block >>> ( \
    d_input, d_output, resX, resY,          \
    inputWidthBytes, outputWidthBytes,      \
    numberEncodedBytes, numberChannels,     \
    *reinterpret_cast<uint3*>(ratioDownScale.data()))
        if      (dataType == tdns::data::TYPE_UCHAR)  CALL_DOWNSAMPLE_SAMPLING(uint8_t );
        else if (dataType == tdns::data::TYPE_USHORT) CALL_DOWNSAMPLE_SAMPLING(uint16_t);
        else if (dataType == tdns::data::TYPE_UINT)   CALL_DOWNSAMPLE_SAMPLING(uint32_t);
        else if (dataType == tdns::data::TYPE_CHAR)   CALL_DOWNSAMPLE_SAMPLING(int8_t  );
        else if (dataType == tdns::data::TYPE_SHORT)  CALL_DOWNSAMPLE_SAMPLING(int16_t );
        else if (dataType == tdns::data::TYPE_INT)    CALL_DOWNSAMPLE_SAMPLING(int32_t );
        else if (dataType == tdns::data::TYPE_FLOAT)  CALL_DOWNSAMPLE_SAMPLING(float   );
        else if (dataType == tdns::data::TYPE_DOUBLE) CALL_DOWNSAMPLE_SAMPLING(double  );
        else throw std::runtime_error("Unsupported data type: " + std::to_string(dataType));
#undef CALL_DOWNSAMPLE_SAMPLING

        // Synchronize to check for any kernel launch errors
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_CHECK_KERNEL_ERROR();

        // Copy back data from destination device memory to output image in CPU memory
        CUDA_SAFE_CALL(cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL(cudaFree(d_input));
        CUDA_SAFE_CALL(cudaFree(d_output));
    }

    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
    Mipmapper::Mipmapper()
    {
        if (!tdns::data::Configuration::get_instance().get_field("BrickSize", _brickEdgeSize))
        {
            //LOGERROR(20, "Unable to get the BrickSize from configuration file. Set default value to 32;");
            _brickEdgeSize = 32;
        }
    }

    //---------------------------------------------------------------------------------------------------
    Mipmapper::~Mipmapper()
    {
        ;
    }

    //---------------------------------------------------------------------------------------------------
    bool Mipmapper::init(tdns::math::Vector3ui &dimension,
                            tdns::math::Vector3ui &ratioDownScale,
                            tdns::data::DataTypeEnum &dataType,
                            uint32_t &numberEncodedBytes,
                            uint32_t &numberChannels,
                            std::string &mipmapDirectory)
    {
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();

        std::string dataTypeStr;
        if (!conf.get_field("DataType", dataTypeStr))
        {
            LOGERROR(20, "Unable to get the DataType from configuration file.");
            throw std::runtime_error("Unable to get the DataType from configuration file.");
        }
        dataType = tdns::data::getDataTypeEnum(dataTypeStr);

        if (!conf.get_field("NumberEncodedBytes", numberEncodedBytes))
        {
            LOGERROR(20, "Unable to get the NumberEncodedBytes from configuration file. Set default value to 1.");
            numberEncodedBytes = 1;
        }

        if (tdns::data::getDataTypeSize(dataType) != numberEncodedBytes)
        {
            LOGERROR(20, "The NumberEncodedBytes is not consistent with the DataType.");
            throw std::runtime_error("The NumberEncodedBytes is not consistent with the DataType.");
        }

        if (!conf.get_field("NumberChannels", numberChannels))
        {
            LOGERROR(20, "Unable to get the NumberChannels from configuration file. Set default value to 1.");
            numberChannels = 1;
        }

        // Get the volume dimensions
        if (!conf.get_field("size_X", dimension[0]))
        {
            LOGERROR(20, "Unable to initialize the Mipmapper. size_X not set in configuration.");
            return false;
        }

        if (!conf.get_field("size_Y", dimension[1]))
        {
            LOGERROR(20, "Unable to initialize the Mipmapper. size_Y not set in configuration.");
            return false;
        }

        if (!conf.get_field("size_Z", dimension[2]))
        {
            //LOGERROR(20, "Unable to initialize the Mipmapper. size_Z not set in configuration.");
            return false;
        }

        // Get the down-scale ratios
        if (!conf.get_field("downScale_X", ratioDownScale[0]))
        {
            LOGERROR(20, "Unable to initialize the Mipmapper. downScale_X not set in configuration.");
            return false;
        }

        if (!conf.get_field("downScale_Y", ratioDownScale[1]))
        {
            LOGERROR(20, "Unable to initialize the Mipmapper. downScale_Y not set in configuration.");
            return false;
        }

        if (!conf.get_field("downScale_Z", ratioDownScale[2]))
        {
            //LOGERROR(20, "Unable to initialize the Mipmapper. downScale_Z not set in configuration.");
            return false;
        }

        std::string volumeDirectory;
        if (!conf.get_field("VolumeDirectory", volumeDirectory))
        {
            LOGERROR(20, "Unable to get the Volume directory from configuration file.");
            return false;
        }
        // create the mimap directory
        mipmapDirectory = volumeDirectory + "/mipmap/";
        
        if (!tdns::common::is_dir(mipmapDirectory))
            tdns::common::create_folder(mipmapDirectory);

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    void Mipmapper::process(tdns::data::MetaData &metaData)
    {
        // Get the full directory + filename
        std::string fileName, workingDirectory;
        if(!tdns::data::Configuration::get_instance().get_field("WorkingDirectory", workingDirectory))
        {
            LOGERROR(20, "Unable to get the directory of the volume. workingDirectory not set in configuration.");
            return;
        }
        if(!tdns::data::Configuration::get_instance().get_field("VolumeFile", fileName))
        {
            LOGERROR(20, "Unable to get the file name of the volume. VolumeFile not set in configuration.");
            return;
        }
        std::string filePath = workingDirectory + tdns::common::get_file_base_name(fileName) + "/" + tdns::common::get_file_name(fileName);

        // Get the file
        std::unique_ptr<tdns::data::AbstractFile> fileUp = tdns::data::FilesManager::get_instance().get_file_from_path(filePath);
        if (!fileUp)
        {
            LOGERROR(20, "Cannot find the file " + filePath);
            return;
        }

        tdns::math::Vector3ui dimension(0);
        tdns::math::Vector3ui ratioDownScale(0);
        uint32_t numberEncodedBytes;
        std::string mipmapDirectory;
        uint32_t numberChannels;
        tdns::data::DataTypeEnum dataType;
        
        // initialization
        if (!init(dimension, ratioDownScale, dataType, numberEncodedBytes, numberChannels, mipmapDirectory))
        {
            LOGERROR(20, "Mipmapper initialization failed !");
            return;
        }

        fileUp->open();

        // Largest dimension of the 3 axis of the initial volume.
        const uint32_t maxDim = std::max(dimension[0] /* FIXME : * ratio voxel en x*/, std::max(dimension[1]/* FIXME : * ratio voxel en y */, dimension[2]/* FIXME : * ratio voxel en z */));
        // Highest level of the pyramid.
        const uint32_t levelMax = static_cast<uint32_t>(std::ceil(std::log(maxDim / _brickEdgeSize) / std::log(2)));

        tdns::math::Vector3ui curentDim = dimension;

        LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "Processing volume " << fileName);
        LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "Mipmap pyramid : " << levelMax << " levels" );

        // Allocation of input and output buffers
        // ----------------------------------------------------------------------
        uint64_t inputSize = MAX_BYTES_SEND_TO_GPU;
        uint64_t outputSize = inputSize / (ratioDownScale[0] * ratioDownScale[1] * ratioDownScale[2]);

        uint8_t *input = nullptr;
        uint8_t *output = nullptr;

        // Pinned memory (non paginable) allocation of the input and the output host buffers
        CUDA_SAFE_CALL(cudaHostAlloc(&input, inputSize, cudaHostAllocDefault || cudaHostAllocWriteCombined));
        CUDA_SAFE_CALL(cudaHostAlloc(&output, outputSize, cudaHostAllocDefault));
        // ----------------------------------------------------------------------

        std::unique_ptr<tdns::data::AbstractFile> fileDown;
        // For each level of the mipmap pyramid
        for (uint64_t level = 1; level <= levelMax; ++level)
        {
            LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "level : " << level << " starting ...");
            
            // Create the new file for the current level of the mipmap pyramid
            std::string fileDownName = "L" + std::to_string(level) + ".raw";
            fileDown = tdns::common::create_unique_ptr<tdns::data::RAWFile>(mipmapDirectory + fileDownName);

            fileDown->open();

            // How many lines can be send to the GPU
            uint32_t nbLines = MAX_BYTES_SEND_TO_GPU / (curentDim[0] * numberEncodedBytes * ratioDownScale[2]);

            // Limit at one complete slide in case of small volumes.
            nbLines = std::min(nbLines, curentDim[1]);

            // For each slices of the current level volume
            for (uint64_t i = 0; i < curentDim[2]; i+=ratioDownScale[2])
            {
                uint64_t nbLinesToRead = nbLines;
                uint64_t linesIndex = 0;

                // As long as there are lines
                while(linesIndex < curentDim[1])
                {
                    uint64_t sizeToRead = nbLinesToRead * curentDim[0] * numberEncodedBytes;
                    uint64_t sizeTowrite = (curentDim[0] / ratioDownScale[0]) * (nbLinesToRead / ratioDownScale[1]) * numberEncodedBytes;

                    // for each slice of the current z down sampling ratio
                    for (uint64_t j = 0; j < ratioDownScale[2]; ++j)
                    {
                        uint64_t sliceAbsolutePosition = std::min<uint64_t>(i+j, curentDim[2]-1) * curentDim[0] * curentDim[1] * numberEncodedBytes;
                        uint64_t cursorPosition = sliceAbsolutePosition + linesIndex * curentDim[0] * numberEncodedBytes;
                        fileUp->set_absolute_cursor_position(cursorPosition);
                        fileUp->read(&input[j * sizeToRead], static_cast<uint32_t>(sizeToRead));
                    }

                    down_scale(output, input, curentDim[0], static_cast<uint32_t>(nbLinesToRead), numberEncodedBytes, numberChannels, dataType, ratioDownScale);

                    fileDown->write(output, static_cast<uint32_t>(sizeTowrite));

                    linesIndex += nbLinesToRead;
                    if (linesIndex + nbLinesToRead > curentDim[1]) nbLinesToRead = curentDim[1] - linesIndex;
                }
            }

            if (curentDim[0] > ratioDownScale[0]) curentDim[0] /= ratioDownScale[0];
            if (curentDim[1] > ratioDownScale[1]) curentDim[1] /= ratioDownScale[1];
            if (curentDim[2] > ratioDownScale[2]) curentDim[2] /= ratioDownScale[2];

            // fileDown becomes fileUp
            fileUp.swap(fileDown);

            LOGINFO(20, tdns::common::log_details::Verbosity::MEDIUM, "level : " << level << " done !");
        }

        CUDA_SAFE_CALL(cudaFreeHost(output));
        CUDA_SAFE_CALL(cudaFreeHost(input));

        fill_metaData(metaData.get_initial_levels(), levelMax, dimension, ratioDownScale);
    }

    //---------------------------------------------------------------------------------------------------
    void Mipmapper::down_scale(uint8_t *output,
                                    const uint8_t *input,
                                    const uint32_t resX,
                                    const uint32_t resY,
                                    const uint32_t numberEncodedBytes,
                                    const uint32_t numberChannels,
                                    tdns::data::DataTypeEnum dataType,
                                    tdns::math::Vector3ui ratioDownScale)
    {
        tdns::preprocessor::down_scale(output, input, resX, resY, numberEncodedBytes, numberChannels, dataType, ratioDownScale);
    }

    //---------------------------------------------------------------------------------------------------
    void Mipmapper::fill_metaData(std::vector<tdns::math::Vector3ui> &initialLevels, uint32_t levels, tdns::math::Vector3ui dimension, tdns::math::Vector3ui ratioDownScale)
    {
        uint32_t sizeX = dimension[0];
        uint32_t sizeY = dimension[1];
        uint32_t sizeZ = dimension[2];

        for (uint32_t i = 0; i <= levels; ++i)
        {
            initialLevels.push_back(tdns::math::Vector3ui(sizeX, sizeY, sizeZ));

            if (sizeX >= ratioDownScale[0]) sizeX /= ratioDownScale[0];
            if (sizeY >= ratioDownScale[1]) sizeY /= ratioDownScale[1];
            if (sizeZ >= ratioDownScale[2]) sizeZ /= ratioDownScale[2];
        }
    }
} // namespace preprocessor
} // namespace tdns
