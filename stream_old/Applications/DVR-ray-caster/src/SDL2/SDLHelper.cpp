#include <SDL2/SDLHelper.hpp>

#include <stdexcept>

#include <SDL2/SDL.h>

namespace tdns
{
namespace graphics
{
    //---------------------------------------------------------------------------------------------------
    void create_sdl_context(uint32_t flags)
    {
        if (SDL_Init(flags) != 0)
        {
            throw std::runtime_error("SDL init error" + std::string(SDL_GetError()));
        }
    }

    //---------------------------------------------------------------------------------------------------
    void quit_sdl()
    {
        SDL_Quit();
    }
} // namespace graphics
} // namespace tdns

