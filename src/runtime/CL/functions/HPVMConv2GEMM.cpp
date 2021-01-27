
#include "arm_compute/runtime/CL/functions/HPVMConv2GEMM.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/SubTensor.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyNativeKernel.h"
#include "support/MemorySupport.h"
#include <cstddef>
#include <memory>

using namespace arm_compute;

inline bool is_central_element_index(size_t K, size_t index)
{
    return index == (K / 2 * (K + 1));
}

TensorInfo create_info(TensorShape shape, DataType data_type)
{
    return TensorInfo(shape, 1, data_type, DataLayout::NHWC);
}

AccumulatingGEMM::AccumulatingGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager))
{
}

AccumulatingGEMM::~AccumulatingGEMM()
{
}

void AccumulatingGEMM::configure(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                                 const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                 const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(),
              input, weights, output,
              conv_info, weights_info,
              dilation, act_info, num_groups);
}

void AccumulatingGEMM::configure(const CLCompileContext &compile_context,
                                 const ICLTensor *input, const ICLTensor *weights, ICLTensor *output,
                                 const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                 const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), weights->info(), output->info(), conv_info, weights_info, dilation, act_info, num_groups));

    // n filters
    M = weights->info()->dimension(0);
    // channels per image/filter
    C = weights->info()->dimension(1);
    // filter size
    K = weights->info()->dimension(2);

    HW = input->info()->dimension(0);

    _filter_tensor.allocator()->init(create_info(TensorShape(C, K * K * M), DataType::F32));
    _output_buffer.allocator()->init(create_info(TensorShape(M, HW), DataType::F32));
    _output_tensor.allocator()->init(create_info(TensorShape(M, HW), DataType::F32));

    _memory_group.manage(&_filter_tensor);
    _memory_group.manage(&_output_buffer);
    _memory_group.manage(&_output_tensor);

    // TODO: reshape [weights] into [_filter_tensor];

    GEMMLHSMatrixInfo lhs_info{};
    lhs_info.m0 = 1;
    lhs_info.k0 = 2;

    GEMMRHSMatrixInfo rhs_info{};
    rhs_info.n0 = 2;
    rhs_info.k0 = lhs_info.k0;

    for(size_t i = 0; i < K * K; i++)
    {
        GEMMKernelInfo kernel_info{};
        kernel_info.m        = M;
        kernel_info.k        = C;
        kernel_info.n        = HW;
        kernel_info.lhs_info = lhs_info;
        kernel_info.rhs_info = rhs_info;
        kernel_info.a_offset = i * M * C;

        auto used_output = is_central_element_index(K, i) ? &_output_tensor : &_output_buffer;

        auto k = arm_compute::support::cpp14::make_unique<CLGEMMMatrixMultiplyNativeKernel>();
        k->configure(compile_context, &_filter_tensor, input, nullptr, used_output, 1, 1, lhs_info, rhs_info, kernel_info);
        _filter_image_gemmkernels.push_back(std::move(k));
    }

    _filter_tensor.allocator()->allocate();
    _output_buffer.allocator()->allocate();
    _output_tensor.allocator()->allocate();
}

Status AccumulatingGEMM::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output,
                                  const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                  const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NCHW);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(weights, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(weights, DataLayout::NCHW);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(2) != weights->dimension(3), "Only square filters supported");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(1) != weights->dimension(1), "Mismatching channels on weights and input");

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);

    return Status{};
}

void AccumulatingGEMM::run()
{
    ARM_COMPUTE_ERROR("AccumulatingGEMM::run() not supported. Use AccumulatingGEMM::run(size_t filter_perforation) instead.");
}

void AccumulatingGEMM::run(size_t filter_perforation)
{
    _memory_group.acquire();

    for(size_t i = 0; i < K * K; i++)
    {
        if(is_central_element_index(K, i) || filter_perforation == 1 || i % filter_perforation == 0)
        {
            CLScheduler::get().enqueue(*_filter_image_gemmkernels[i]);

            // TODO: accumulate results
        }
    }

    _memory_group.release();
}

void AccumulatingGEMM::prepare()
{
}