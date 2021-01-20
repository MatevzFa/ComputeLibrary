#ifndef ARM_COMPUTE_LOGGING_ANDROID_PRINTER_H
#define ARM_COMPUTE_LOGGING_ANDROID_PRINTER_H

#include "arm_compute/core/utils/logging/IPrinter.h"

#include <iostream>

namespace arm_compute
{
namespace logging
{
/** Std Printer */
class AndroidPrinter final : public Printer
{
private:
    // Inherited methods overridden:
    void print_internal(const std::string &msg) override;
};
} // namespace logging
} // namespace arm_compute
#endif /* ARM_COMPUTE_LOGGING_ANDROID_PRINTER_H */
