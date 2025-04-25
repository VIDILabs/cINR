#pragma once

#include <GcCore/libCommon/CppNorm.hpp>

extern "C"
{
    void TDNS_API start_logger(char* path_name);

    void TDNS_API end_logger();
}