#ifndef ARM_COMPUTE_HPVMTYPES_H_N
#define ARM_COMPUTE_HPVMTYPES_H_N

#include "arm_compute/core/Error.h"

#include <cstddef>

enum HPVMConvApproxPerfMode
{
    ROW,
    COL,
    FILTER,
    NONE,
};

struct HPVMConvApproxInfo
{
    HPVMConvApproxPerfMode mode;
    size_t                 perf_start;
    size_t                 perf_every;

    HPVMConvApproxInfo()
        : mode(HPVMConvApproxPerfMode::NONE), perf_start(0), perf_every(0)
    {
    }

    HPVMConvApproxInfo(HPVMConvApproxPerfMode mode, size_t start, size_t every)
        : mode(mode),
          perf_start(start),
          perf_every(every)
    {
    }

    // HPVMConvApproxInfo &operator=(const HPVMConvApproxInfo &info) = default;

    static HPVMConvApproxInfo from_hpvm(int row, int col, int skip_every, int offset)
    {
        if(row > 1)
            return HPVMConvApproxInfo(HPVMConvApproxPerfMode::ROW, offset, row);

        if(col > 1)
            return HPVMConvApproxInfo(HPVMConvApproxPerfMode::COL, offset, col);

        if(skip_every > 1)
            return HPVMConvApproxInfo(HPVMConvApproxPerfMode::FILTER, offset, skip_every);

        ARM_COMPUTE_ERROR("unreachable");
    }
};

#endif