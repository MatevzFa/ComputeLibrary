#ifndef ARM_COMPUTE_HPVM_CONVAPPROX_H
#define ARM_COMPUTE_HPVM_CONVAPPROX_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "src/core/CL/kernels/HPVMFilterPerfKernel.h"
#include "src/core/CL/kernels/HPVMIm2ColPerfRowKernel.h"
#include <cstddef>
#include <memory>

namespace arm_compute
{
enum HPVMConvApproxPerfMode
{
    ROW,
    COL,
    FILTER,
};

struct HPVMConvApproxInfo
{
    const HPVMConvApproxPerfMode mode;
    const size_t                 perf_start;
    const size_t                 perf_every;

    HPVMConvApproxInfo(HPVMConvApproxPerfMode mode, size_t start, size_t every)
        : mode(mode),
          perf_start(start),
          perf_every(every)
    {
    }

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

class HPVMConvApprox : public IFunction
{
public:
    HPVMConvApprox();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    HPVMConvApprox(const HPVMConvApprox &) = delete;

    /** Default move constructor */
    HPVMConvApprox(HPVMConvApprox &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    HPVMConvApprox &operator=(const HPVMConvApprox &) = delete;

    /** Default move assignment operator */
    HPVMConvApprox &operator=(HPVMConvApprox &&) = default;

    /**Default destructor */
    ~HPVMConvApprox();

    void configure(ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                   const PadStrideInfo &conv_info, const HPVMConvApproxInfo &perf_info);

    void configure(const CLCompileContext &compile_context,
                   ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                   const PadStrideInfo &conv_info, const HPVMConvApproxInfo &perf_info);

    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output,
                           const PadStrideInfo &conv_info, const HPVMConvApproxInfo &perf_info);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLTensor _im2col_tensor;
    CLTensor _filter_tensor;
    CLTensor _gemm_output;

    std::unique_ptr<HPVMIm2ColPerfRowKernel> _im2col_kernel;
    std::unique_ptr<HPVMFilterPerfKernel>    _filterperf_kernel;
    std::unique_ptr<CLGEMM>                  _gemm;
};
}; // namespace arm_compute

#endif