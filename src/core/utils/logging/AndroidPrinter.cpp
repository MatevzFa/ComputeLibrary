#include "arm_compute/core/utils/logging/AndroidPrinter.h"

#include <android/log.h>

using namespace arm_compute::logging;

void AndroidPrinter::print_internal(const std::string &msg)
{
    __android_log_write(ANDROID_LOG_INFO, "ARM-ComputeLibrary-Log", msg.c_str());
}
