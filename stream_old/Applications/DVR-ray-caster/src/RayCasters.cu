#include <RayCasters_helpers.hpp>

// CUDA math helper
#include <helper_math.h>

namespace tdns
{
namespace graphics
{
    __constant__ float4x3 d_invModelViewMatrix;  // inverse model*view matrix

    //---------------------------------------------------------------------------------------------
    __device__ float3   TDNS_API operator*(float* matrix, float3 v);
    __device__ float3   TDNS_API mul(const float4x3 &M, const float3 &v);
    __device__ float4   TDNS_API mul(const float4x3 &M, const float4 &v);
    __device__ int      TDNS_API intersectBox(float3 p, float3 d, float3 boxmin, float3 boxmax, float *tnear, float *tfar);
    __device__ uint32_t TDNS_API rgbaFloatToInt(float4 &rgba);
    __device__ uint32_t TDNS_API compute_LOD(float voxelDistance, uint32_t levelMax, uint3 *levelsSize, float3 *LODBrickSize);    
    __device__ void     TDNS_API print_bb_edges(const float3 &bboxMin, const float3 &bboxMax, const float3 &position, float4 &color);

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
                            size_t seed)
    {
        // 2D Thread ID
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x > screenSize.x || y > screenSize.y) return;

        // Transform the 2D screen coords into [-1; 1] range
        float u = ((x + 0.5f) / (float) screenSize.x) * 2.0f - 1.0f;
        float v = ((y + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f;

        // calculate eye ray in world space
        float3 origin = make_float3(mul(d_invModelViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
        float3 direction = normalize(make_float3(u, v, -2.f));
        direction = mul(d_invModelViewMatrix, direction);

        // find intersection with box
        float tnear, tfar;
        int32_t intersected = intersectBox(origin, direction, bboxMin, bboxMax, &tnear, &tfar);
        
        const float3 bgColor = {0.f, 0.f, 0.f};
        float4 finalColor = {0.f, 0.f, 0.f, 0.f};

        if (!intersected)
        {
            finalColor.x = bgColor.x;
            finalColor.y = bgColor.y;
            finalColor.z = bgColor.z;
        }
        else
        {
            if (tnear < 0.f) tnear = 0.f;          // clamp to near plane

            float t = tnear;
            float3 position = origin + direction * tnear;
            float3 step;
            // float3 step = direction * tstep;
            float sample;
            float4 color = {0.f, 0.f, 0.f, 0.f};
            float3 texturePosition = {0.f, 0.f, 0.f};
            float3 normalizedPosition = {0.f, 0.f, 0.f};

            // Print the bouding box edges in red
            // print_bb_edges(bboxMin, bboxMax, position, finalColor);
            
            // march along ray from front to back, accumulating color
            while (finalColor.w < 0.95f && t < tfar)
            {
                // Transform the [-1; 1] world position into a [0; 0.99[ range volume coords for the texture.
                texturePosition = position * 0.495f + 0.495f;
                texturePosition.z = 1 - texturePosition.z;

                // sampling
                normalizedPosition = clamp(texturePosition, 0.f, 0.99f);
                uint32_t lod = compute_LOD(t, 4, levelsSize, LODBrickSize);
                tdns::gpucache::VoxelStatus voxelStatus = manager.template get_normalized<float>(lod, normalizedPosition, sample);

                // Handle Unmapped and Empty bricks
                if (voxelStatus == tdns::gpucache::VoxelStatus::Empty || voxelStatus == tdns::gpucache::VoxelStatus::Unmapped)
                {
                    float3 brickSize = LODBrickSize[lod];
                    t += brickSize.x;
                    if (t > tfar)
                        break;
                    position += direction * brickSize.x;
                    continue;
                }

                // classification
                color = tex1D<float4>(tfTex, sample);

                color.w = 1.f - powf(1.f - color.w, LODStepSize[lod] / tstep);
                // color.w = 1.f - powf(1.f - color.w, tstep / tstep);

                // skip transparent samples
                if (color.w > 0.001f)
                {
                    // pre-multiply alpha
                    color.x *= color.w;
                    color.y *= color.w;
                    color.z *= color.w;

                    // front-to-back blending operator
                    finalColor += color * (1.0f - finalColor.w);
                }

                // add the step size according to the LOD of the current sample
                t += LODStepSize[lod];
                // t += tstep;
                
                // use the step size according to the LOD of the current sample
                step = direction * LODStepSize[lod];
                // step = direction * tstep;
                position += step;
                    
            }
        }
        // Brightness
        // finalColor *= 1.5f;

        uint32_t i = y * screenSize.x + x;
        pixelBuffer[i] = rgbaFloatToInt(finalColor);
    }

    //---------------------------------------------------------------------------------------------
    template <>
    __global__ void RayCast(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<ushort1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            size_t seed)
    {
        // 2D Thread ID
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= renderScreenWidth || y >= screenSize.y) return;

        // Transform the 2D screen coords into [-1; 1] range
        float u = (((x + 0.5f) / (float) renderScreenWidth) * 2.0f - 1.0f) * tan(fov / 2);
        float v = (((y + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f) * tan(fov / 2);

        // calculate eye ray in world space
        float3 origin = make_float3(mul(d_invModelViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
        float3 direction = normalize(make_float3(u, v, -2.f)); // u, v : pixel position in the image plan, -2.f : distance from camera to image plan
        direction = mul(d_invModelViewMatrix, direction);  // then apply the model view matrix

        // find intersection with box
        float tnear, tfar;
        int32_t intersected = intersectBox(origin, direction, bboxMin, bboxMax, &tnear, &tfar);
        
        const float4 bgColor = {0.f, 0.f, 0.f, 0.f};
        float4 finalColor = {0.0f, 0.0f, 0.0f, 0.0f};

        if (!intersected)
        {
            finalColor.x = bgColor.x;
            finalColor.y = bgColor.y;
            finalColor.z = bgColor.z;
            finalColor.w = bgColor.w;
        }
        else
        {   
            if (tnear < 0.0f) tnear = 0.0f;          // clamp to near plane

            float t = tnear;
            float3 position = origin + direction * tnear;
            float3 step;
            // float3 step = direction * tstep;
            float sample;
            float4 color = {0.f, 0.f, 0.f, 0.f};
            float3 texturePosition = {0.f, 0.f, 0.f};
            float3 normalizedPosition = {0.f, 0.f, 0.f};

            // Print the bouding box edges in red
            print_bb_edges(bboxMin, bboxMax, position, finalColor);
            
            // march along ray from front to back, accumulating color
            for (int32_t i = 0; i < steps; ++i)
            {
                // Transform the [-1; 1] world position into a [0; 0.99[ range volume coords for the cache texture
                texturePosition = position * 0.495f + 0.495f;
                texturePosition.z = 1 - texturePosition.z;

                // sampling
                normalizedPosition = clamp(texturePosition, 0.f, 0.99f);
                uint32_t lod = compute_LOD(t, 4, levelsSize, LODBrickSize);
                tdns::gpucache::VoxelStatus voxelStatus = manager.get_normalized<float>(lod, normalizedPosition, sample);
                
                // Handle Unmapped and Empty bricks
                if (voxelStatus == tdns::gpucache::VoxelStatus::Empty || voxelStatus == tdns::gpucache::VoxelStatus::Unmapped)
                {
                    float3 brickSize = LODBrickSize[lod];
                    t += brickSize.x;
                    if (t > tfar)
                        break;
                    position += direction * brickSize.x;
                    continue;
                }

                // classification
                color = tex1D<float4>(tfTex, sample);

                color.w = 1.f - powf(1.f - color.w, LODStepSize[lod] / tstep);
                // color.w = 1.f - powf(1.f - color.w, tstep / tstep);

                // skip transparent samples
                if (color.w > 0.001f)
                {
                    // pre-multiply alpha
                    color.x *= color.w;
                    color.y *= color.w;
                    color.z *= color.w;

                    // front-to-back blending operator
                    finalColor += color * (1.0f - finalColor.w);
                }

                // add the step size according to the LOD of the current sample
                t += LODStepSize[lod];
                // t += tstep;
                
                // exit early if opaque or if we are outside the volume
                if (finalColor.w > 0.95f || t > tfar)
                    break;

                // use the step size according to the LOD of the current sample
                step = direction * LODStepSize[lod];
                // step = direction * tstep;
                position += step;
            }
        }
        // Brightness
        // finalColor *= 1.5f;

        uint32_t i = y * screenSize.x + x;
        pixelBuffer[i] = rgbaFloatToInt(finalColor);
    }

    //---------------------------------------------------------------------------------------------
    template <>
    __global__ void RayCast(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<float1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            size_t seed)
    {
        // 2D Thread ID
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x > screenSize.x || y > screenSize.y) return;

        // Transform the 2D screen coords into [-1; 1] range
        float u = ((x + 0.5f) / (float) screenSize.x) * 2.0f - 1.0f;
        float v = ((y + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f;

        // calculate eye ray in world space
        float3 origin = make_float3(mul(d_invModelViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
        float3 direction = normalize(make_float3(u, v, -2.f));
        direction = mul(d_invModelViewMatrix, direction);

        // find intersection with box
        float tnear, tfar;
        int32_t intersected = intersectBox(origin, direction, bboxMin, bboxMax, &tnear, &tfar);
        
        const float3 bgColor = {0.f, 0.f, 0.f};
        float4 finalColor = {0.f, 0.f, 0.f, 0.f};

        if (!intersected)
        {
            finalColor.x = bgColor.x;
            finalColor.y = bgColor.y;
            finalColor.z = bgColor.z;
        }
        else
        {
            if (tnear < 0.f) tnear = 0.f;          // clamp to near plane

            float t = tnear;
            float3 position = origin + direction * tnear;
            float3 step;
            // float3 step = direction * tstep;
            float sample;
            float4 color = {0.f, 0.f, 0.f, 0.f};
            float3 texturePosition = {0.f, 0.f, 0.f};
            float3 normalizedPosition = {0.f, 0.f, 0.f};

            // Print the bouding box edges in red
            // print_bb_edges(bboxMin, bboxMax, position, finalColor);
            
            // march along ray from front to back, accumulating color
            while (finalColor.w < 0.95f && t < tfar)
            {
                // Transform the [-1; 1] world position into a [0; 0.99[ range volume coords for the texture.
                texturePosition = position * 0.495f + 0.495f;
                texturePosition.z = 1 - texturePosition.z;

                // sampling
                normalizedPosition = clamp(texturePosition, 0.f, 0.99f);
                uint32_t lod = compute_LOD(t, 5, levelsSize, LODBrickSize);
                tdns::gpucache::VoxelStatus voxelStatus = manager.get_normalized<float>(lod, normalizedPosition, sample);

                // Handle Unmapped and Empty bricks
                if (voxelStatus == tdns::gpucache::VoxelStatus::Empty || voxelStatus == tdns::gpucache::VoxelStatus::Unmapped)
                {
                    float3 brickSize = LODBrickSize[lod];
                    t += brickSize.x;
                    if (t > tfar)
                        break;
                    position += direction * brickSize.x;
                    continue;
                }

                // classification
                color = tex1D<float4>(tfTex, sample);

                color.w = 1.f - powf(1.f - color.w, LODStepSize[lod] / tstep);

                // skip transparent samples
                if (color.w > 0.001f)
                {
                    // pre-multiply alpha
                    color.x *= color.w;
                    color.y *= color.w;
                    color.z *= color.w;

                    // front-to-back blending operator
                    finalColor += color * (1.0f - finalColor.w);
                }

                // add the step size according to the LOD of the current sample
                t += LODStepSize[lod];
                
                // use the step size according to the LOD of the current sample
                step = direction * LODStepSize[lod];
                position += step;
                    
            }
        }
        // Brightness
        // finalColor *= 1.5f;

        uint32_t i = y * screenSize.x + x;
        pixelBuffer[i] = rgbaFloatToInt(finalColor);
    }

    //---------------------------------------------------------------------------------------------
    template <typename T>
    void declare_RayCast(tdns::gpucache::K_CacheManager<T> manager) 
    {
        throw std::runtime_error("should never be called");
        dim3 gridDim;
        dim3 blockDim;
        RayCast<T><<<gridDim, blockDim>>>(
            nullptr, cudaTextureObject_t{}, uint2{}, 0, 0.f, float3{}, float3{}, 0, 0.f, 
            manager, nullptr, nullptr, nullptr, nullptr, 0);
    }
    template void declare_RayCast(tdns::gpucache::K_CacheManager<uchar1>  manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<ushort1> manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<uint1>   manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<char1>   manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<short1>  manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<int1>    manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<float1>  manager);
    template void declare_RayCast(tdns::gpucache::K_CacheManager<double1> manager);

    // DEVICE FUNC

    //---------------------------------------------------------------------------------------------
    inline __device__ float3 operator*(float* matrix, float3 v)
    {
        float x = matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z;
        float y = matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z;
        float z = matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z;

        return make_float3(x, y, z);
    }

    //---------------------------------------------------------------------------------------------
    // transform vector by matrix (no translation)
    inline __device__ float3 mul(const float4x3 &M, const float3 &v)
    {
        float3 r;
        r.x = dot(v, make_float3(M.m[0]));
        r.y = dot(v, make_float3(M.m[1]));
        r.z = dot(v, make_float3(M.m[2]));
        return r;
    }

    //---------------------------------------------------------------------------------------------
    // transform vector by matrix with translation
    inline __device__ float4 mul(const float4x3 &M, const float4 &v)
    {
        float4 r;
        r.x = dot(v, M.m[0]);
        r.y = dot(v, M.m[1]);
        r.z = dot(v, M.m[2]);
        r.w = 1.0f;
        return r;
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ int intersectBox(float3 p, float3 d, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
    {
        // compute intersection of ray with all six bbox planes
        float3 invR = make_float3(1.0f) / d;
        float3 tbot = invR * (boxmin - p);
        float3 ttop = invR * (boxmax - p);
    
        // re-order intersections to find smallest and largest on each axis
        float3 tmin = fminf(ttop, tbot);
        float3 tmax = fmaxf(ttop, tbot);
    
        // find the largest tmin and the smallest tmax
        float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
        float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
    
        *tnear = largest_tmin;
        *tfar = smallest_tmax;
    
        return smallest_tmax > largest_tmin;
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ uint32_t rgbaFloatToInt(float4 &rgba)
    {
        rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
        rgba.y = __saturatef(rgba.y);
        rgba.z = __saturatef(rgba.z);
        rgba.w = __saturatef(rgba.w);

        return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ uint32_t compute_LOD(float voxelDistance, uint32_t levelMax, uint3 *levelsSize, float3 *LODBrickSize)
    {
        // float fov = 70.f;
        // float near = 0.1f;
        // int32_t i;

        // for (i = levelMax; i >= 0; --i)
        // {
        //     float V = LODBrickSize[i].x / levelsSize[i].x;
        //     if (V * (near / voxelDistance) < 1.f / fov)
        //         break;
        // }

        // return i;

        uint32_t level = (voxelDistance + 4.f) / 5.f;
        if (level > levelMax) level = levelMax;
        return level;
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ void print_bb_edges(const float3 &bboxMin, const float3 &bboxMax, const float3 &position, float4 &color)
    {
        // The epsilon determine the thickness of the printed edges. The higher it is, the thicker the line will be.
        float epsilon = 0.02f;
        // Print the bouding box edges in red
        if ((position.x < (bboxMin.x + epsilon) || position.x > (bboxMax.x - epsilon)) && ((position.y < (bboxMin.y + epsilon) || position.y >            (bboxMax.y - epsilon)) || (position.z < (bboxMin.z + epsilon) || position.z > (bboxMax.z - epsilon))) ||
            (position.y < (bboxMin.y + epsilon) || position.y > (bboxMax.y - epsilon)) && ((position.x < (bboxMin.x + epsilon) || position.x > (bboxMax.x - epsilon)) || (position.z < (bboxMin.z + epsilon) || position.z > (bboxMax.z - epsilon))) ||
            (position.z < (bboxMin.z + epsilon) || position.z > (bboxMax.z - epsilon)) && ((position.x < (bboxMin.x + epsilon) || position.x > (bboxMax.x - epsilon)) || (position.y < (bboxMin.y + epsilon) || position.y > (bboxMax.y - epsilon))))
            
            color = make_float4(1.f, 0.f, 0.f, 1.f);
    }


    //---------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------------------
    void update_CUDA_inv_view_model_matrix(const float *invViewMatrix)
    {
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_invModelViewMatrix, invViewMatrix, 3 * sizeof(float4)));
    }

    //---------------------------------------------------------------------------------------------
    glm::mat3 update_CUDA_inv_view_model_matrix_offline(const glm::mat4 &viewMatrix, const glm::mat4 &volumeMatrix)
    {
        glm::mat4 invModelViewMat = volumeMatrix * viewMatrix;
        // glm::mat4 invModelViewMat = glm::inverse(modelViewMat); // if the model-view matrix (ignoring the translation) is orthonormal (i.e. consists only of rotations), then its inverse is just its transpose, which is much cheaper to compute. https://www.opengl.org/discussion_boards/showthread.php/195803-Getting-World-Space-Eye-Vector

        // send to the GPU the inverse model-vue matrix
        float4 invModelViewMatrix[3];
        invModelViewMatrix[0] = make_float4(invModelViewMat[0].x, invModelViewMat[1].x, invModelViewMat[2].x, invModelViewMat[3].x);
        invModelViewMatrix[1] = make_float4(invModelViewMat[0].y, invModelViewMat[1].y, invModelViewMat[2].y, invModelViewMat[3].y);
        invModelViewMatrix[2] = make_float4(invModelViewMat[0].z, invModelViewMat[1].z, invModelViewMat[2].z, invModelViewMat[3].z);

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_invModelViewMatrix, invModelViewMatrix, 3 * sizeof(float4)));

        // return the upper left 3x3 matrix
        return glm::mat3(invModelViewMat);
    }

    cudaArray *d_volumeArray;
    //---------------------------------------------------------------------------------------------
    void create_CUDA_transfer_function(cudaTextureObject_t &tfTexture, cudaArray *d_transferFuncArray)
    {
        // Specify texture
        struct cudaResourceDesc resDescTex;
        memset(&resDescTex, 0, sizeof(resDescTex));
        resDescTex.resType = cudaResourceTypeArray;
        resDescTex.res.array.array = d_transferFuncArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.normalizedCoords = 1;
        texDesc.addressMode[0] = cudaAddressModeClamp;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&tfTexture, &resDescTex, &texDesc, NULL));
    }

    //---------------------------------------------------------------------------------------------
    void update_CUDA_transfer_function(tdns::math::Vector4f *transferFunctionPtr, uint32_t transferFunctionWidth, cudaArray *d_transferFuncArray)
    {
        int32_t pitch = transferFunctionWidth * sizeof(float4);
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, transferFunctionPtr, pitch, pitch, 1, cudaMemcpyHostToDevice));
    }
    //---------------------------------------------------------------------------------------------
    void destroy_CUDA_transfer_function()
    {
        // CUDA_SAFE_CALL(cudaFreeArray(d_transferFuncArray));
    }
} // namespace graphics
} // namespace tdns
