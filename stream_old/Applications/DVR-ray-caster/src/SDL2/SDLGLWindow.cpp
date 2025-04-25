#include <SDL2/SDLGLWindow.hpp>

#include <stdexcept>

#include <SDL2/SDL.h>

namespace tdns
{
namespace graphics
{
    //---------------------------------------------------------------------------------------------------
    SDLGLWindow::SDLGLWindow(const std::string &title, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t flags)
    {
        SDL_Window *window = SDL_CreateWindow(title.c_str(), x, y, w, h, flags | SDL_WINDOW_OPENGL);
        if (!window) std::runtime_error("Error in SDLGLWindow window creation [" + std::string(SDL_GetError()) + "].");
        
        SDL_GLContext glcontext = SDL_GL_CreateContext(window);
        if (!glcontext) std::runtime_error("Error in SDLGLWindow glContext creation [" + std::string(SDL_GetError()) + "].");

        _window = window;
        _glContext = glcontext;

        // disable Vsync
        SDL_GL_SetSwapInterval(0);
    }

    //---------------------------------------------------------------------------------------------------
    SDLGLWindow::~SDLGLWindow()
    {
        SDL_GL_DeleteContext(_glContext);
        if (_window) SDL_DestroyWindow(_window);
    }

    //---------------------------------------------------------------------------------------------------
    void SDLGLWindow::display()
    {
        SDL_GL_SwapWindow(_window);
    }

    //---------------------------------------------------------------------------------------------------
    void SDLGLWindow::set_title(const std::string &title)
    {
        SDL_SetWindowTitle(_window, title.c_str());
    }

    //---------------------------------------------------------------------------------------------------
    bool SDLGLWindow::is_full_screen() const
    {
        return SDL_GetWindowFlags(_window) && SDL_WINDOW_FULLSCREEN_DESKTOP;
    }
    
    //---------------------------------------------------------------------------------------------------
    void SDLGLWindow::set_full_screen()
    {
        SDL_SetWindowFullscreen(_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
    }

    //---------------------------------------------------------------------------------------------------
    void SDLGLWindow::unset_full_screen()
    {
        SDL_SetWindowFullscreen(_window, 0);
    }

    //---------------------------------------------------------------------------------------------------
    bool SDLGLWindow::has_focus() const
    {
        return SDL_GetWindowFlags(_window) && SDL_WINDOW_MOUSE_FOCUS;
    }

    //---------------------------------------------------------------------------------------------------
    tdns::math::Vector2i SDLGLWindow::size() const
    {
        tdns::math::Vector2i xy;
        SDL_GetWindowSize(_window, &xy[0], &xy[1]);
        return xy;
    }

    //---------------------------------------------------------------------------------------------------
    SDL_Window* SDLGLWindow::get_window()
    {
        return _window;
    }

    //---------------------------------------------------------------------------------------------------
    SDL_GLContext SDLGLWindow::get_gl_context()
    {
        return _glContext;
    }

} // namespace graphics
} // namespace tdns
