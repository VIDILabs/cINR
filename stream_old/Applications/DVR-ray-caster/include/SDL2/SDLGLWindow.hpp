#pragma once

#include <cstdint>
#include <string>

#include <SDL2/SDL.h>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace graphics
{
    class TDNS_API SDLGLWindow
    {
    public:
        SDLGLWindow(const std::string &title, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t flags);

        ~SDLGLWindow();

        void display();

        void set_title(const std::string &title);

        bool is_full_screen() const;

        void set_full_screen();

        void unset_full_screen();

        bool has_focus() const;

        tdns::math::Vector2i size() const;

        SDL_Window* get_window();

        SDL_GLContext get_gl_context();

    private:

        SDL_Window *_window;
        SDL_GLContext _glContext;
    };
} // namespace graphics
} // namespace tdns
