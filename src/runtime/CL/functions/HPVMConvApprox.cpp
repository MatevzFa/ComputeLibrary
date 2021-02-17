#include "arm_compute/runtime/CL/functions/HPVMConvApprox.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/IMemoryGroup.h"

#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/kernels/HPVMFilterPerfKernel.h"
#include "src/core/CL/kernels/HPVMIm2ColPerfRowKernel.h"
#include "support/MemorySupport.h"

#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "ARMComputeLibrary", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ARMComputeLibrary", __VA_ARGS__)

#include <sstream>

using namespace arm_compute;

HPVMConvApprox::HPVMConvApprox()
{
}

HPVMConvApprox::~HPVMConvApprox()
{
}

void HPVMConvApprox::configure(ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                               const PadStrideInfo &conv_info, const HPVMConvApproxInfo &perf_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, output, conv_info, perf_info);
}

void HPVMConvApprox::configure(const CLCompileContext &compile_context,
                               ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                               const PadStrideInfo &conv_info, const HPVMConvApproxInfo &perf_info)
{
    // ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), weights->info(), output->info(), conv_info, perf_info));

    _im2col_kernel     = support::cpp14::make_unique<HPVMIm2ColPerfRowKernel>();
    _filterperf_kernel = support::cpp14::make_unique<HPVMFilterPerfKernel>();

    auto n_idx = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::BATCHES);
    auto c_idx = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::CHANNEL);
    auto h_idx = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::HEIGHT);
    auto w_idx = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::WIDTH);

    auto channels = input->info()->dimension(c_idx);
    auto batches  = input->info()->dimension(n_idx);
    auto nfilters = weights->info()->dimension(n_idx);

    Size2D kernel_shape(weights->info()->dimension(w_idx), weights->info()->dimension(h_idx));

    // Determine im2col shape
    TensorShape im2col_tensor_shape;
    switch(perf_info.mode)
    {
        case HPVMConvApproxPerfMode::ROW:
        {
            auto h_eff = input->info()->dimension(h_idx) - input->info()->dimension(h_idx) / perf_info.perf_every;
            im2col_tensor_shape.set(0, kernel_shape.area() * channels);
            im2col_tensor_shape.set(1, input->info()->dimension(w_idx) * h_eff);
            im2col_tensor_shape.set(2, batches);
        }
        break;

        case HPVMConvApproxPerfMode::COL:
        {
            auto w_eff = input->info()->dimension(w_idx) - input->info()->dimension(w_idx) / perf_info.perf_every;
            im2col_tensor_shape.set(0, kernel_shape.area() * channels);
            im2col_tensor_shape.set(1, w_eff * input->info()->dimension(h_idx));
            im2col_tensor_shape.set(2, batches);
        }
        break;

        case HPVMConvApproxPerfMode::FILTER:
        {
            im2col_tensor_shape.set(0, (kernel_shape.area() - kernel_shape.area() / perf_info.perf_every) * channels);
            im2col_tensor_shape.set(1, input->info()->dimension(w_idx) * input->info()->dimension(h_idx));
            im2col_tensor_shape.set(2, batches);
        }
        break;
    }
    _im2col_tensor.allocator()->init(input->info()->clone()->set_tensor_shape(im2col_tensor_shape));

    // Determine filter shape
    TensorShape filter_tensor_shape;
    filter_tensor_shape.set(0, nfilters);
    filter_tensor_shape.set(1, _im2col_tensor.info()->dimension(w_idx));
    _filter_tensor.allocator()->init(weights->info()->clone()->set_tensor_shape(filter_tensor_shape));

    // Init kernels
    switch(perf_info.mode)
    {
        case HPVMConvApproxPerfMode::ROW:
        {
            _im2col_kernel->configure(input, &_im2col_tensor, kernel_shape, conv_info, false,
                                      HPVMIm2ColPerfInfo::perfrow(perf_info.perf_start, perf_info.perf_every));
            _filterperf_kernel->configure(weights, &_filter_tensor,
                                          HPVMFilterPerfInfo(perf_info.perf_start, perf_info.perf_every));
        }
        break;

        case HPVMConvApproxPerfMode::COL:
        {
            ARM_COMPUTE_ERROR("unreachable");
        }
        break;

        case HPVMConvApproxPerfMode::FILTER:
        {
            LOGE("input          %ld %ld %ld %ld", input->info()->dimension(0), input->info()->dimension(1), input->info()->dimension(2), input->info()->dimension(3));
            LOGE("kernel_shape   %ld %ld", kernel_shape.width, kernel_shape.height);
            LOGE("_im2col_tensor %ld %ld %ld %ld", _im2col_tensor.info()->dimension(0), _im2col_tensor.info()->dimension(1), _im2col_tensor.info()->dimension(2), _im2col_tensor.info()->dimension(3));
            _im2col_kernel->configure(input, &_im2col_tensor, kernel_shape, conv_info, false,
                                      HPVMIm2ColPerfInfo::perffilter(perf_info.perf_start, perf_info.perf_every));
            _filterperf_kernel->configure(weights, &_filter_tensor,
                                          HPVMFilterPerfInfo(perf_info.perf_start, perf_info.perf_every));
        }
        break;
    }

    // Init GEMM
    _gemm_output.allocator()->init(output->info()->set_tensor_shape({
        nfilters,
        _im2col_tensor.info()->dimension(h_idx),
        batches,
    }));

    _gemm = support::cpp14::make_unique<CLGEMM>();

    _gemm->configure(&_im2col_tensor, &_filter_tensor, nullptr, &_gemm_output, 1, 1);
}

Status HPVMConvApprox::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output,
                                const PadStrideInfo &conv_info, const HPVMConvApproxInfo &perf_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NCHW);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(weights, DataLayout::NCHW);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(output, DataLayout::NCHW);

    ARM_COMPUTE_RETURN_ERROR_ON(perf_info.mode == HPVMConvApproxPerfMode::COL);

    return Status{};
}

void HPVMConvApprox::prepare()
{
}

void HPVMConvApprox::run()
{
    _im2col_tensor.allocator()->allocate();
    _filter_tensor.allocator()->allocate();
    _gemm_output.allocator()->allocate();

    CLScheduler::get().enqueue(*_im2col_kernel);
    CLScheduler::get().sync();

    {
        _im2col_tensor.map();
        std::ostringstream out;
        _im2col_tensor.print(out);
        LOGI("_im2col_tensor\n%s", out.str().c_str());
        _im2col_tensor.unmap();
    }

    CLScheduler::get().enqueue(*_filterperf_kernel);
    CLScheduler::get().sync();

    {
        _filter_tensor.map();
        std::ostringstream out;
        _filter_tensor.print(out);
        LOGI("_filter_tensor\n%s", out.str().c_str());
        _filter_tensor.unmap();
    }
    _gemm->run();
    CLScheduler::get().sync();

    {
        _gemm_output.map();
        std::ostringstream out;
        _gemm_output.print(out);
        LOGI("_gemm_output\n%s", out.str().c_str());
        _gemm_output.unmap();
    }

    _im2col_tensor.allocator()->free();
    _filter_tensor.allocator()->free();
    _gemm_output.allocator()->free();
}
