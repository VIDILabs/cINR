#include <SDL2/SDLConstext.hpp>

#include <stdexcept>
#include <string>

#include <SDL2/SDL.h>

namespace tdns
{
namespace graphics
{
    //---------------------------------------------------------------------------------------------------
    SDLContext::SDLContext(uint32_t flags)
    {
        if (SDL_Init(flags) != 0)
        {
            throw std::runtime_error("SDL init error" + std::string(SDL_GetError()));
        }
    }

    //---------------------------------------------------------------------------------------------------
    SDLContext::~SDLContext()
    {
        SDL_Quit();
    }
} // namespace graphics
} // namespace tdns