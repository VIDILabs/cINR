#pragma once

#if TDNS_OS == TDNS_OS_WINDOWS
    #include <winsock2.h>
    #ifndef _WIN32_WINNT
        #define _WIN32_WINNT 0x0501  /* Windows XP. */
    #endif
    #include <Ws2tcpip.h>    
#else
// Linux use int
    #include <cstdint>
    #define SOCKET int32_t
    #include <arpa/inet.h>
    #include <sys/types.h>    
    #include <sys/socket.h>
    #include <unistd.h>
#endif