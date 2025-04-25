#pragma once

/**
* Compilation flag description, must be provided by the build system:
*
* /!\ All flags and values are prefixed with "TDNS_" in order to avoid name collision with alread existing system.
*
*  - Operatong system description, as of now the following OS are supported:
*      TDNS_WINDOWS  : Various microsoft operating system
*      TDNS_LINUx    : Various linux operating system
*      TDNS_MAC      : Varios mac operating system
*
*  - Compilation mode:
*      TDNS_DEBUG
*      TDNS_RELEASE
*/

// OS
#define TDNS_OS_WINDOWS       1000
#define TDNS_OS_LINUX         1001
#define TDNS_OS_MAC           1002

// Compilation Mode
#define TDNS_MODE_DEBUG       1100
#define TDNS_MODE_RELEASE     1101
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------

// Operatin system
#if defined TDNS_WINDOWS
#   define TDNS_OS TDNS_OS_WINDOWS
#elif defined TDNS_LINUX
#   define TDNS_OS TDNS_OS_LINUX
#elif defined TDNS_MAC
#   define TDNS_OS TDNS_OS_MAC
#else
#   undef TDNS_OS
static_assert(false, "No TDNS_XXX flag (describing the os type) provided by the build system!");
#endif
//

// Compilation mode
#if defined TDNS_DEBUG
#   define TDNS_MODE TDNS_MODE_DEBUG
#elif defined TDNS_RELEASE
#   define TDNS_MODE TDNS_MODE_RELEASE
#else
#   undef TDNS_MODE
static_assert(false, "The TDNS_MODE flag (describing the compilation mode) cannot be computed with values provided by the build system!");
#endif
//

// Export / import macro
#if !defined TDNS_STATIC
#   if TDNS_OS == TDNS_OS_WINDOWS
#       define TDNS_API_EXPORT __declspec(dllexport)
#       define TDNS_API_IMPORT __declspec(dllimport)
#       ifdef _MSC_VER
// For Visual C++ compilers, we also need to turn off this annoying C4251 warning
#           pragma warning(disable: 4251)
#       endif
#   elif TDNS_OS == TDNS_OS_LINUX
#       if __GNUC__ >= 4
        // GCC 4 has special keywords for showing/hidding symbols.
#           define TDNS_API_EXPORT __attribute__ ((__visibility__ ("default")))
#           define TDNS_API_IMPORT __attribute__ ((__visibility__ ("default")))
#       else
        // Nothing !
#           define TDNS_API_EXPORT
#           define TDNS_API_IMPORT
#       endif
#   endif
#else
// Static => no needs to use import / export macro.
#   define TDNS_API_EXPORT
#   define TDNS_API_IMPORT
#endif

// Export / import mode
#if defined TDNS_EXPORT
#   define TDNS_API TDNS_API_EXPORT
#else
#   define TDNS_API TDNS_API_IMPORT
#endif