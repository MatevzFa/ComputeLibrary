#include "arm_compute/runtime/CL/functions/HPVMConvApprox.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/HPVMTypes.h"
#include "arm_compute/runtime/IMemoryGroup.h"

#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/kernels/HPVMFilterPerfKernel.h"
#include "src/core/CL/kernels/HPVMIm2ColPerfRowKernel.h"
#include "src/core/CL/kernels/HPVMInterpolateKernel.h"
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
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), weights->info(), output->info(), conv_info, perf_info));

    _im2col_kernel     = support::cpp14::make_unique<HPVMIm2ColPerfRowKernel>();
    _filterperf_kernel = support::cpp14::make_unique<HPVMFilterPerfKernel>();

    _perf_info = perf_info;

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

        default:
            ARM_COMPUTE_ERROR("unreachable");
    }
    _im2col_tensor.allocator()->init(input->info()->clone()->set_tensor_shape(im2col_tensor_shape));

    // Determine filter shape
    TensorShape filter_tensor_shape;
    filter_tensor_shape.set(0, nfilters);
    filter_tensor_shape.set(1, _im2col_tensor.info()->dimension(w_idx));

    TensorInfo filter_tensor_info(
        filter_tensor_shape, 1,
        weights->info()->data_type(), weights->info()->data_layout());
    _filter_tensor.allocator()->init(filter_tensor_info);

    // Init kernels
    switch(perf_info.mode)
    {
        case HPVMConvApproxPerfMode::ROW:
        {
            _im2col_kernel->configure(input, &_im2col_tensor, kernel_shape, conv_info, false,
                                      HPVMIm2ColPerfInfo::perfrow(perf_info.perf_start, perf_info.perf_every));
            _filterperf_kernel->configure(weights, &_filter_tensor, HPVMFilterPerfInfo(0, 0));
        }
        break;

        case HPVMConvApproxPerfMode::COL:
        {
            ARM_COMPUTE_ERROR("unreachable");
        }
        break;

        case HPVMConvApproxPerfMode::FILTER:
        {
            // LOGE("input          %ld %ld %ld %ld", input->info()->dimension(0), input->info()->dimension(1), input->info()->dimension(2), input->info()->dimension(3));
            // LOGE("weights        %ld %ld %ld %ld", weights->info()->dimension(0), weights->info()->dimension(1), weights->info()->dimension(2), weights->info()->dimension(3));
            // LOGE("kernel_shape   %ld %ld", kernel_shape.width, kernel_shape.height);
            // LOGE("_im2col_tensor %ld %ld %ld %ld", _im2col_tensor.info()->dimension(0), _im2col_tensor.info()->dimension(1), _im2col_tensor.info()->dimension(2), _im2col_tensor.info()->dimension(3));
            // LOGE("_filter_tensor %ld %ld %ld %ld", _filter_tensor.info()->dimension(0), _filter_tensor.info()->dimension(1), _filter_tensor.info()->dimension(2), _filter_tensor.info()->dimension(3));
            _im2col_kernel->configure(input, &_im2col_tensor, kernel_shape, conv_info, false,
                                      HPVMIm2ColPerfInfo::perffilter(perf_info.perf_start, perf_info.perf_every));
            _filterperf_kernel->configure(weights, &_filter_tensor,
                                          HPVMFilterPerfInfo(0, 2));
        }
        break;

        default:
            ARM_COMPUTE_ERROR("unreachable");
    }

    // Init GEMM
    TensorInfo gemm_output_info(*output->info());
    gemm_output_info.set_tensor_shape({
        nfilters,
        _im2col_tensor.info()->dimension(h_idx),
        batches,
    });
    _gemm_output.allocator()->init(gemm_output_info);

    TensorInfo gemm_output_transposed_info(*output->info());
    gemm_output_transposed_info.set_tensor_shape({
        _im2col_tensor.info()->dimension(h_idx),
        nfilters,
        batches,
    });
    _gemm_output_transposed.allocator()->init(gemm_output_transposed_info);

    _gemm = support::cpp14::make_unique<CLGEMM>();
    _gemm->configure(&_im2col_tensor, &_filter_tensor, nullptr, &_gemm_output, 1, 1);

    _transpose.configure(&_gemm_output, &_gemm_output_transposed);

    if(_perf_info.mode == HPVMConvApproxPerfMode::FILTER)
    {
        _reshape.configure(&_gemm_output_transposed, output);
    }
    else if(_perf_info.mode == HPVMConvApproxPerfMode::ROW)
    {
        _interpolate_kernel = support::cpp14::make_unique<HPVMInterpolateKernel>();
        _interpolate_kernel->configure(&_gemm_output_transposed, output, perf_info);
    }
    else
    {
        ARM_COMPUTE_ERROR("unimplemented");
    }
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
    bool do_print = false;

    _im2col_tensor.allocator()->allocate();
    _filter_tensor.allocator()->allocate();
    _gemm_output.allocator()->allocate();
    _gemm_output_transposed.allocator()->allocate();

    CLScheduler::get().enqueue(*_im2col_kernel);

    if(do_print)
    {
        _im2col_tensor.map();
        std::ostringstream out;
        _im2col_tensor.print(out);
        LOGI("_im2col_tensor\n%s", out.str().c_str());
        _im2col_tensor.unmap();
    }

    CLScheduler::get().enqueue(*_filterperf_kernel);

    if(do_print)
    {
        _filter_tensor.map();
        std::ostringstream out;
        _filter_tensor.print(out);
        LOGI("_filter_tensor\n%s", out.str().c_str());
        _filter_tensor.unmap();
    }

    _gemm->run();

    if(do_print)
    {
        _gemm_output.map();
        std::ostringstream out;
        _gemm_output.print(out);
        LOGI("_gemm_output\n%s", out.str().c_str());
        _gemm_output.unmap();
    }

    _transpose.run();

    if(do_print)
    {
        _gemm_output_transposed.map();
        std::ostringstream out;
        _gemm_output_transposed.print(out);
        LOGI("_gemm_output_transposed\n%s", out.str().c_str());
        _gemm_output_transposed.unmap();
    }

    if(_perf_info.mode == HPVMConvApproxPerfMode::FILTER)
    {
        _reshape.run();
    }
    else if(_perf_info.mode == HPVMConvApproxPerfMode::ROW)
    {
        CLScheduler::get().enqueue(*_interpolate_kernel);
    }
    else
    {
        ARM_COMPUTE_ERROR("unimplemented");
    }

    _im2col_tensor.allocator()->free();
    _filter_tensor.allocator()->free();
    _gemm_output.allocator()->free();
    _gemm_output_transposed.allocator()->free();
}
