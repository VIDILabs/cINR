#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/cuda/libGPUCache/K_CacheManager.hpp>

#include <helper_math.h>

namespace tdns
{
namespace graphics
{
    typedef struct
    {
        float4 m[3];
    } float4x3;


    template <typename T>
    __global__ void RayCast(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<T> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            size_t seed);

    __device__ tdns::gpucache::VoxelStatus TDNS_API get_sample(  tdns::gpucache::K_CacheManager<ushort1> &manager,
                                                        uint32_t lod,
                                                        const float3 &normalizedPosition,
                                                        float &sample);
                                                        
    void TDNS_API update_CUDA_inv_view_model_matrix(const float *invViewMatrix);
    glm::mat3 TDNS_API update_CUDA_inv_view_model_matrix_offline(const glm::mat4 &viewMatrix, const glm::mat4 &volumeMatrix);
    void TDNS_API create_CUDA_volume(uint32_t texWidth, uint32_t texHeight, uint32_t texDepth, uint8_t *dataPtr);
    void TDNS_API create_CUDA_transfer_function(cudaTextureObject_t &tfTexture, cudaArray *d_transferFuncArray);
    void TDNS_API update_CUDA_transfer_function(tdns::math::Vector4f *transferFunctionPtr, uint32_t transferFunctionWidth, cudaArray *d_transferFuncArray);
    void TDNS_API destroy_CUDA_volume();
    void TDNS_API destroy_CUDA_transfer_function();

} // namespace graphics
} // namespace tdns
