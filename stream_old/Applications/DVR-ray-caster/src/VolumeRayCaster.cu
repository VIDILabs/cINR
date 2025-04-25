#include <VolumeRayCaster.hpp>

#include <chrono>
#include <vector>
#include <cmath>
#include <memory>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

#include <Shader.hpp>
#include <Display_helper.hpp>
#include <Camera.hpp>
#include <RayCasters_helpers.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_sdl2.h>
#include <imgui/imgui_impl_opengl3.h>

#include <SDL2/SDLHelper.hpp>


namespace tdns
{
namespace graphics
{
    //---------------------------------------------------------------------------------------------
    template<typename T>
    void display(   SDLGLWindow &window,
                    Shader &shader,
                    tdns::math::Vector2ui &screenSize,
                    tdns::math::Vector3f &bboxmin, tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    TransferFunction &tf,
                    tdns::data::MetaData &volumeData,
                    tdns::gpucache::CacheManager<T> *manager,
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize,
                    std::vector<float> &histo);

    template<typename T>
    void get_frame( uint32_t *d_pixelBuffer,
                    cudaTextureObject_t &tfTex,
                    const tdns::math::Vector2ui &screenSize,
                    const tdns::math::Vector3f &bboxmin, const tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    tdns::gpucache::CacheManager<T> *manager,
                    const Camera &camera,
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize);

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void display_volume_raycaster(tdns::gpucache::CacheManager<T> *manager, tdns::data::MetaData &volumeData)
    {
        // Get the needed fiel in the configuration file
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        tdns::math::Vector2ui screenSize;
        conf.get_field("ScreenWidth", screenSize[0]);
        conf.get_field("ScreenHeight", screenSize[1]);
        std::string volumeDirectory;
        conf.get_field("VolumeDirectory", volumeDirectory);
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);
        
        // Init SDL
        create_sdl_context(SDL_INIT_EVERYTHING);

        // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100
        const char* glsl_version = "#version 100";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
        // GL 3.2 Core + GLSL 150
        const char* glsl_version = "#version 150";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

        // From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
        SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

        SDLGLWindow sdlWindow("3DNS", SDL_WINDOWPOS_UNDEFINED, 100, screenSize[0], screenSize[1], SDL_WINDOW_RESIZABLE);
        
        // Init GLEW
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if (GLEW_OK != err)
            std::cout << "Failed to initialize GLEW : " << glewGetErrorString(err) << std::endl;
        std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

        glViewport(0, 0, screenSize[0], screenSize[1]);

        // Init ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        ImGui::StyleColorsDark();
        ImGui_ImplSDL2_InitForOpenGL(sdlWindow.get_window(), NULL);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Init GLSL shader
        Shader shader("shaders/shader.vs", "shaders/shader.fs");

        // Load the pre-computed histogram of the volume
        std::vector<float> &histo = volumeData.get_histo();
        size_t histoSize = histo.size();

        // Create and init a transfer function
        TransferFunction tf = TransferFunction(128);

        // Bounding box
        tdns::math::Vector3f bboxmin(-1.f, -1.f, -1.f);
        tdns::math::Vector3f bboxmax(1.f, 1.f, 1.f);
        tdns::math::Vector3f bboxSize(bboxmax[0] - bboxmin[0], bboxmax[1] - bboxmin[1], bboxmax[2] - bboxmin[2]);
        float bboxSizeMin = std::min(bboxSize[0], std::min(bboxSize[1], bboxSize[2]));
        
        // Sampling rate : in order to accurately reconstruct the original signal from the discrete data we need to take at least two samples per voxel.
        float marchingStep = bboxSizeMin / 256.f;

        // Configure parameters with the number of LOD to give to the kernel ray caster
        /* --------------------------------------------------------------------------------------------------------------------- */
        uint32_t nbLevels = volumeData.nb_levels();
        uint3 *levelsSize = reinterpret_cast<uint3*>(volumeData.get_initial_levels().data());
        std::vector<float3> invLevelsSize(nbLevels);
        std::vector<float3> LODBrickSize(nbLevels);
        std::vector<float> LODStepSize(nbLevels);

        for (size_t i = 0; i < nbLevels; ++i)
        {
            invLevelsSize[i] = make_float3( bboxSizeMin / static_cast<float>(levelsSize[i].x),
                                            bboxSizeMin / static_cast<float>(levelsSize[i].y),
                                            bboxSizeMin / static_cast<float>(levelsSize[i].z));

            LODBrickSize[i] = make_float3((bboxSize[0] / levelsSize[i].x) * brickSize, (bboxSize[0] / levelsSize[i].y) * brickSize, (bboxSize[0] / levelsSize[i].z) * brickSize);
            // Determine the sampling step size according to the LOD in order to have 2 samples per voxels.
            LODStepSize[i] = (bboxSizeMin / (std::min(levelsSize[i].x, std::min(levelsSize[i].y, levelsSize[i].z)))) / 2.f;
        }

        uint3 *d_levelsSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_levelsSize,  nbLevels * sizeof(uint3)));
        CUDA_SAFE_CALL(cudaMemcpy(d_levelsSize, levelsSize, nbLevels * sizeof(uint3), cudaMemcpyHostToDevice));
        float3 *d_invLevelsSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_invLevelsSize,  nbLevels * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMemcpy(d_invLevelsSize, invLevelsSize.data(), nbLevels * sizeof(float3), cudaMemcpyHostToDevice));
        float3 *d_LODBrickSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_LODBrickSize,  nbLevels * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMemcpy(d_LODBrickSize, LODBrickSize.data(), nbLevels * sizeof(float3), cudaMemcpyHostToDevice));
        float *d_LODStepSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_LODStepSize,  nbLevels * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_LODStepSize, LODStepSize.data(), nbLevels * sizeof(float), cudaMemcpyHostToDevice));
        /* --------------------------------------------------------------------------------------------------------------------- */

        /*********************** CALL THE DISPLAY FUNCTION ***********************/
        display(sdlWindow, shader, screenSize, bboxmin, bboxmax, marchingStep, tf, volumeData, manager, d_levelsSize, d_invLevelsSize, d_LODBrickSize, d_LODStepSize, histo);
    }

    template  void display_volume_raycaster(tdns::gpucache::CacheManager<uchar1>  *manager, tdns::data::MetaData &volumeData);
    template  void display_volume_raycaster(tdns::gpucache::CacheManager<ushort1> *manager, tdns::data::MetaData &volumeData);
    template  void display_volume_raycaster(tdns::gpucache::CacheManager<uint1>   *manager, tdns::data::MetaData &volumeData);
    template  void display_volume_raycaster(tdns::gpucache::CacheManager<char1>   *manager, tdns::data::MetaData &volumeData);
    template  void display_volume_raycaster(tdns::gpucache::CacheManager<short1>  *manager, tdns::data::MetaData &volumeData);
    template  void display_volume_raycaster(tdns::gpucache::CacheManager<int1>    *manager, tdns::data::MetaData &volumeData);
    template  void display_volume_raycaster(tdns::gpucache::CacheManager<float1>  *manager, tdns::data::MetaData &volumeData);
    // template  void display_volume_raycaster(tdns::gpucache::CacheManager<double1> *manager, tdns::data::MetaData &volumeData);

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void display(   SDLGLWindow &window,
                    Shader &shader,
                    tdns::math::Vector2ui &screenSize,
                    tdns::math::Vector3f &bboxmin, tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    TransferFunction &tf,
                    tdns::data::MetaData &volumeData,
                    tdns::gpucache::CacheManager<T> *manager,
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize,
                    std::vector<float> &histo)
    {
        // Init pixel buffer object and CUDA-OpenGL interoperability ressources
        uint32_t pbo, vbo, vao, texture;
        struct cudaGraphicsResource *cuda_pbo_dest_resource;
        init_GL_ressources_raycaster(pbo, texture, &cuda_pbo_dest_resource, screenSize);
        init_mesh_raycaster(&vbo, &vao);
        
        // Add the camera
        Camera camera;

        // Model and View Matrix
        // glm::mat4 modelMatrix = glm::mat4(1.f); // identity matrix
        // glm::mat4 viewMatrix;
        
        // define the background color
        ImVec4 bgColor = ImVec4(0.1f, 0.1f, 0.1f, 1.f);
            
        // Get the size of the histogram of the volume
        size_t histoSize = histo.size();

        // Map the openGL pixel buffer to accessible CUDA space memory
        uint32_t *d_pixelBuffer;
        size_t num_bytes;
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&d_pixelBuffer, &num_bytes, cuda_pbo_dest_resource));

        // CUDA transfer function
        cudaArray *d_transferFuncArray;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        CUDA_SAFE_CALL(cudaMallocArray(&d_transferFuncArray, &channelDesc, 128, 1));
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, tf.get_samples_data(), 128 * sizeof(float4), 128 * sizeof(float4), 1, cudaMemcpyHostToDevice));
        cudaTextureObject_t tfTex;
	    create_CUDA_transfer_function(tfTex, d_transferFuncArray);

        // Enable and configure openGL alpha transparency 
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // model-view matrix
        float3 viewRotation;
        float3 viewTranslation = make_float3(0.0, 0.0, -15.0f);
        float invViewMatrix[12];
        
        bool run = true;

        // main loop
        while (run)
        {    
            // Clear the colorbuffer
	        glClearColor(bgColor.x, bgColor.y, bgColor.z, bgColor.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            update_CUDA_transfer_function(reinterpret_cast<tdns::math::Vector4f*>(tf.get_samples_data()), 128, d_transferFuncArray);

            // Activate shader
            shader.use();
            
            updateViewMatrix(invViewMatrix, viewRotation, viewTranslation);
            // viewMatrix = camera.GetViewMatrix();

            // update_CUDA_inv_view_model_matrix(viewMatrix, modelMatrix);
            update_CUDA_inv_view_model_matrix(invViewMatrix);
            get_frame(d_pixelBuffer, tfTex, screenSize, bboxmin, bboxmax, marchingStep, manager, camera, d_levelsSize,d_invLevelsSize, d_LODBrickSize, d_LODStepSize);

            // Bind VAO
            glBindVertexArray(vao);

            // Activate the texture and give the location to the shader
            glActiveTexture(GL_TEXTURE);
            glBindTexture(GL_TEXTURE_2D, texture);
            glUniform1i(glGetUniformLocation(shader.Program, "tex"), 0);

            // Download texture from destination PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenSize[0], screenSize[1], GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

            glDrawArrays(GL_TRIANGLES, 0, 6);

            render_gui(window, bgColor, bboxmin, bboxmax, histo, histoSize, tf);

            // Swap buffer
            window.display();

            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);

            // Update the caches manager
            manager->update();

            // Compute cache completude
            std::vector<float> completude;
            manager->completude(completude);

            // Print window title
            std::string title = "3DNS - [Cache used " + std::to_string(completude[0] * 100.f) + "%]";
            window.set_title(title);

            handle_event(window, viewRotation, viewTranslation, marchingStep, run);
        }

        /********  Cleanup ********/

        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));        

        // destroy_CUDA_volume();
        destroy_CUDA_transfer_function();

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture);

        CUDA_SAFE_CALL(cudaFree(d_levelsSize));
        CUDA_SAFE_CALL(cudaFree(d_invLevelsSize));
        CUDA_SAFE_CALL(cudaFree(d_LODBrickSize));
        CUDA_SAFE_CALL(cudaFree(d_LODStepSize));

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();

        quit_sdl();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void get_frame( uint32_t *d_pixelBuffer,
                    cudaTextureObject_t &tfTex,
                    const tdns::math::Vector2ui &screenSize,
                    const tdns::math::Vector3f &bboxmin, const tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    tdns::gpucache::CacheManager<T> *manager,
                    const Camera &camera, 
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize)
    {
        dim3 gridDim = dim3((screenSize[0] % 16 != 0) ? (screenSize[0] / 16 + 1) : (screenSize[0] / 16), (screenSize[1] % 16 != 0) ? (screenSize[1] / 16 + 1) : (screenSize[1] / 16));
        dim3 blockDim(16, 16);

        uint32_t renderScreenWidth = screenSize[0];

        // float theta = tan(camera.get_fov() / 2.f * M_PI / 180.0f);

        // cudaProfilerStart();
        RayCast<<<gridDim, blockDim>>>(
            d_pixelBuffer,
            tfTex,
            *reinterpret_cast<const uint2*>(screenSize.data()),
            renderScreenWidth,
            camera.get_fov(),
            *reinterpret_cast<const float3*>(bboxmin.data()), *reinterpret_cast<const float3*>(bboxmax.data()),
            1024, marchingStep, //1000, 0.0015f    sampleMax, stepSize
            manager->to_kernel_object(),
            d_invLevelsSize,
            d_levelsSize,
            d_LODBrickSize,
            d_LODStepSize,
            time(0));
        // cudaProfilerStop();

#if TDNS_MODE == TDNS_MODE_DEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif

        CUDA_CHECK_KERNEL_ERROR();
    }
} // namespace graphics
} // namespace tdns