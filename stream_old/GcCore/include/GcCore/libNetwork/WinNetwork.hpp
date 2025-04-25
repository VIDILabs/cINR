#pragma once

#include <cstdint>

#include <GcCore/libCommon/CppNorm.hpp>

namespace tdns
{
namespace network
{
    /**
    * @brief Convert IPv4 and IPv6 addresses from text to binary form
    *
    * @param[in]    af   
    * @param[in]    src   
    * @param[in]    dst   
    *
    * @return True if 
    */
#if TDNS_OS == TDNS_OS_WINDOWS
    bool TDNS_API inet_pton(int32_t af, int8_t *src, void *dst);
#endif

} // namespace network
} // namespace tdns