#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <memory>

#include <GL/glew.h>
#include <SDL2/SDL.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <GcCore/libCommon/CppNorm.hpp>

#include <TransferFunction.hpp>
#include <imgui/imgui.h>

#include <SDL2/SDLGLWindow.hpp>

namespace tdns
{
namespace graphics
{
    void TDNS_API init_GL_ressources_raycaster(uint32_t &pbo, uint32_t &texture, cudaGraphicsResource **cuda_pbo_dest_resource, const tdns::math::Vector2ui &screenSize);
    void TDNS_API init_mesh_raycaster(uint32_t *vbo, uint32_t *vao);
    void TDNS_API updateViewMatrix(float *invViewMatrix, const float3 &viewRotation, const float3 &viewTranslation);

    bool TDNS_API render_gui(SDLGLWindow &sdlWindow, ImVec4 &clear_color, tdns::math::Vector3f &bboxmin, tdns::math::Vector3f &bboxmax, const std::vector<float> &histo, size_t histoSize, TransferFunction &tf);
    
    bool TDNS_API handle_event(SDLGLWindow &window, float3 &viewRotation, float3 &viewTranslation, float &marchingStep, bool &run);
    bool TDNS_API handle_keyboard_event(SDL_Event &event, float &marchingStep, float3 &viewRotation, float3 &viewTranslation);
    bool TDNS_API handle_mouse_event(SDL_Event &event, float3 &viewRotation, float3 &viewTranslation);

} // namespace graphics
} // namespace tdns