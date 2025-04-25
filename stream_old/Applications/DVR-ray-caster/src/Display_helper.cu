#include <Display_helper.hpp>

#include <GcCore/cuda/libCommon/CudaError.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_sdl2.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace tdns
{
namespace graphics
{
    //---------------------------------------------------------------------------------------------
    void init_GL_ressources_raycaster(uint32_t &pbo, uint32_t &texture, cudaGraphicsResource **cuda_pbo_dest_resource, const tdns::math::Vector2ui &screenSize)
    {
        // initialize the PBO for transferring data from CUDA to openGL
        uint32_t num_texels = screenSize[0] * screenSize[1];
        uint32_t size_tex_data =  num_texels * 4 * sizeof(GLfloat);  // RGBA
        std::vector<uint8_t> data(size_tex_data);

        // create the pixel buffer object
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_ARRAY_BUFFER, pbo);
        glBufferData(GL_ARRAY_BUFFER, size_tex_data, data.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
                
        // Register the pixel buffer object to the CUDA space
        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(cuda_pbo_dest_resource, pbo, cudaGraphicsRegisterFlagsNone));

        // create the texture
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        // Set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // define the size of the texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenSize[0], screenSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Activate depth test for the Z-buffer
        glEnable(GL_DEPTH_TEST);
    }

    //---------------------------------------------------------------------------------------------
    void init_mesh_raycaster(uint32_t *vbo, uint32_t *vao)
    {
        // Two triangles (+ texture coordinates) for a square 
        float vertices[] = {
	    -1.f, -1.f, -0.5f,  0.0f, 0.0f,
	     1.f, -1.f, -0.5f,  1.0f, 0.0f,
	     1.f,  1.f, -0.5f,  1.0f, 1.0f,
	     1.f,  1.f, -0.5f,  1.0f, 1.0f,
	    -1.f,  1.f, -0.5f,  0.0f, 1.0f,
	    -1.f, -1.f, -0.5f,  0.0f, 0.0f,
	    };

        // Generate a vertex buffer object and a vertex attribute object
        glGenVertexArrays(1, vao);
        glGenBuffers(1, vbo);
        glBindVertexArray(*vao);
        
        // give the array containing the vertices of the two triangles to the VBO
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)0);
        glEnableVertexAttribArray(0);

        // Texture attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Unbind the VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // Unbind the VAO
        glBindVertexArray(0);
    }

    //---------------------------------------------------------------------------------------------
    void updateViewMatrix(float *invViewMatrix, const float3 &viewRotation, const float3 &viewTranslation)
    {
        // use OpenGL to build view matrix
        GLfloat modelView[16];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
        glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
        glPopMatrix();

        invViewMatrix[0] = modelView[0];
        invViewMatrix[1] = modelView[4];
        invViewMatrix[2] = modelView[8];
        invViewMatrix[3] = modelView[12];
        invViewMatrix[4] = modelView[1];
        invViewMatrix[5] = modelView[5];
        invViewMatrix[6] = modelView[9];
        invViewMatrix[7] = modelView[13];
        invViewMatrix[8] = modelView[2];
        invViewMatrix[9] = modelView[6];
        invViewMatrix[10] = modelView[10];
        invViewMatrix[11] = modelView[14];
    }

    //---------------------------------------------------------------------------------------------
    bool render_gui(SDLGLWindow &sdlWindow, ImVec4 &clear_color, tdns::math::Vector3f &bboxmin, tdns::math::Vector3f &bboxmax, const std::vector<float> &histo, size_t histoSize, TransferFunction &tf)
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Params");
        ImGui::SetWindowSize({400,700}, ImGuiCond_FirstUseEver);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::ColorEdit3("Bagground color", (float*)&clear_color);
        ImGui::Text("Crop the volume:");
        bool crop = false;
        crop ^= ImGui::SliderFloat("X-min", &bboxmin[0], -1.0f, 1.0f);
        ImGui::SliderFloat("Y-min", &bboxmin[1], -1.0f, 1.0f);
        ImGui::SliderFloat("Z-min", &bboxmin[2], -1.0f, 1.0f);
        ImGui::SliderFloat("X-max", &bboxmax[0], -1.0f, 1.0f);
        ImGui::SliderFloat("Y-max", &bboxmax[1], -1.0f, 1.0f);
        ImGui::SliderFloat("Z-max", &bboxmax[2], -1.0f, 1.0f);
        ImGui::Text("Transfer function:");
        tf.draw();
        ImGui::End();

        // imgui render
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        return crop;
    }

    //---------------------------------------------------------------------------------------------
    /*                                            EVENTS                                         */
    //---------------------------------------------------------------------------------------------
    bool handle_event(SDLGLWindow &window, float3 &viewRotation, float3 &viewTranslation, float &marchingStep, bool &run)
    {
        bool handleEvent = false;
        SDL_PumpEvents();
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) || ImGui::GetIO().WantCaptureMouse)
            {
                ImGui_ImplSDL2_ProcessEvent(&event);
                return false;
            }

            switch (event.type)
            {
                case SDL_QUIT:
                    run = false;
                    return true;
                case SDL_KEYDOWN:
                    run = handle_keyboard_event(event, marchingStep, viewRotation, viewTranslation);
                    handleEvent = true;
                    break;
                case SDL_MOUSEMOTION:
                    handleEvent = handle_mouse_event(event, viewRotation, viewTranslation);
                    break;
                case SDL_WINDOWEVENT:
                {
                    switch (event.window.event)
                    {
                        case SDL_WINDOWEVENT_SIZE_CHANGED:  // or SDL_WINDOWEVENT_RESIZED ?
                        {
                            glViewport(0, 0, window.size()[0], window.size()[1]);
                        }
                        break;
                    }
                }
                break;
                // case SDL_MOUSEBUTTONDOWN :
                //     marchingStep *= 2;
                //     break;
                // case SDL_MOUSEBUTTONUP :
                //     marchingStep /= 2;
                //     break;
                case SDL_MOUSEWHEEL:
                {
                    float delta = event.wheel.y;
                    // model = glm::translate(model, glm::vec3(0.f, 0.f, -delta));

                    // glm::vec3 cameraPosition = camera.get_position();
                    // camera.set_position(glm::vec3(cameraPosition[0], cameraPosition[1], cameraPosition[2] + (delta / 4)));
                    
                    viewTranslation.z += delta / 10.0f;
                    handleEvent = true;
                }
                break;
                default:
                    break;
            }
        }

        return handleEvent;
    }

    //---------------------------------------------------------------------------------------------------
    bool handle_keyboard_event(SDL_Event &event, float &marchingStep, float3 &viewRotation, float3 &viewTranslation)
    {
        switch (event.key.keysym.sym)
        {
        case SDLK_ESCAPE:
            return false;
        case SDLK_KP_PLUS:
        {
            // viewTranslation.z += 2.f;
            // viewTranslation.z += 0.08f;
            viewTranslation.z += (0.005f * std::abs(viewTranslation.z));
            break;
        }
        case SDLK_KP_MINUS:
        {
            // viewTranslation.z -= 2.f;
            // viewTranslation.z -= 0.08f;
            viewTranslation.z -= (0.005f * std::abs(viewTranslation.z));
            break;
        }
        case SDLK_UP:
        {
            viewRotation.x += 1.f;
            break;
        }
        case SDLK_DOWN:
        {
            viewRotation.x -= 1.f;
            break;
        }
        case SDLK_RIGHT:
        {
            viewRotation.y += 1.f;
            break;
        }
        case SDLK_LEFT:
        {
            viewRotation.y -= 1.f;
            break;
        }
        case SDLK_PAGEUP:
        {
            marchingStep += 0.0001f;
            std::cout << "Marching step = " << marchingStep << std::endl;
            break;
        }
        case SDLK_PAGEDOWN:
        {
            marchingStep -= 0.0001f;
            std::cout << "Marching step = " << marchingStep << std::endl;
            break;
        }
        default:
            break;
        }
        
        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool handle_mouse_event(SDL_Event &event, float3 &viewRotation, float3 &viewTranslation)
    {
        float x = event.motion.xrel;
        float y = event.motion.yrel;

        if (x == 0 && y == 0) return false;

        bool handleEvent = false;

        // Turn the object
        if (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
            viewRotation.x -= y;
            viewRotation.y += x;
            // if (x != 0)
                // model = glm::rotate(model, glm::radians(x), glm::vec3(0.f, 1.f, 0.f));
 
            // if (y != 0)
                // model = glm::rotate(model, glm::radians(-y), glm::vec3(1.f, 0.f, 0.f));

            handleEvent = true;
        }

        // Moove the object
        if (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_RIGHT))
        {
            viewTranslation.x += x / 100.0f;
            viewTranslation.y += y / 100.0f;
            // x /= 100.f;
            // y /= 100.f;

            // model = glm::translate(model, glm::vec3(-x, -y, 0.f));

        //     // camera.set_position(glm::vec3(
        //     //     camera.get_position().x + x,
        //     //     camera.get_position().y - y,
        //     //     camera.get_position().z));

        //     // camera.set_target(glm::vec3(
        //     //     camera.get_target().x + x,
        //     //     camera.get_target().y - y,
        //     //     camera.get_target().z));

            handleEvent = true;            
        }

        return handleEvent;   
    }
} // namespace graphics
} // namespace tdns