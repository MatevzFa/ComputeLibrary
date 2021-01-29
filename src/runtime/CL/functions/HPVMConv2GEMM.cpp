
#include "arm_compute/runtime/CL/functions/HPVMConv2GEMM.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/logging/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/SubTensor.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyNativeKernel.h"
#include "support/MemorySupport.h"

#include <cstddef>
#include <memory>
#include <sstream>

#include "android/log.h"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "ARMComputeLibrary", __VA_ARGS__)

using namespace arm_compute;

inline bool is_central_element_index(size_t KK, size_t index)
{
    size_t _K = sqrt(KK);
    return index == (_K / 2 * (_K + 1));
}

void log_dims(const char *name, ITensorInfo *info)
{
    LOGI("[%s %ld %ld %ld %ld]",
         name,
         info->dimension(0), info->dimension(1), info->dimension(2), info->dimension(3));
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

void AccumulatingGEMM::configure(const ICLTensor *input, ICLTensor *weights, ICLTensor *output,
                                 const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                 const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(),
              input, weights, output,
              conv_info, weights_info,
              dilation, act_info, num_groups);
}

void AccumulatingGEMM::configure(const CLCompileContext &compile_context,
                                 const ICLTensor *input, ICLTensor *weights, ICLTensor *output,
                                 const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                 const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), weights->info(), output->info(), conv_info, weights_info, dilation, act_info, num_groups));

    C  = weights->info()->dimension(0);
    M  = weights->info()->dimension(1);
    KK = weights->info()->dimension(2);

    HW = input->info()->dimension(0);

    LOGI("C %ld", C);
    LOGI("M %ld", M);
    LOGI("KK %ld", KK);
    LOGI("HW %ld", HW);

    log_dims("input", input->info());
    log_dims("weights", weights->info());

    _output_buffer.allocator()->init(create_info(TensorShape(HW, M), DataType::F32));
    _output_tensor.allocator()->init(create_info(TensorShape(HW, M), DataType::F32));

    log_dims("_output_buffer", _output_buffer.info());
    log_dims("_output_tensor", _output_tensor.info());

    _memory_group.manage(&_output_buffer);
    _memory_group.manage(&_output_tensor);

    GEMMLHSMatrixInfo lhs_info{};
    lhs_info.m0 = 1;
    lhs_info.k0 = 2;

    GEMMRHSMatrixInfo rhs_info{};
    rhs_info.n0 = 2;
    rhs_info.k0 = lhs_info.k0;

    for(size_t i = 0; i < KK; i++)
    {
        auto filters_view = support::cpp14::make_unique<CLSubTensor>(weights, TensorShape(C, M, 1), Coordinates(0, 0, i));
        _subtensors.push_back(std::move(filters_view));

        auto used_output = is_central_element_index(KK, i) ? &_output_tensor : &_output_buffer;

        GEMMKernelInfo kernel_info{};
        kernel_info.m        = M;
        kernel_info.k        = C;
        kernel_info.n        = HW;
        kernel_info.lhs_info = lhs_info;
        kernel_info.rhs_info = rhs_info;

        auto k = arm_compute::support::cpp14::make_unique<CLGEMMMatrixMultiplyNativeKernel>();
        k->configure(compile_context, _subtensors[i].get(), input, nullptr, used_output, 1, 1, lhs_info, rhs_info, kernel_info);
        _filter_image_gemmkernels.push_back(std::move(k));
    }

    _output_buffer.allocator()->allocate();
    _output_tensor.allocator()->allocate();
}

Status AccumulatingGEMM::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output,
                                  const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                  const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    auto hw = input->dimension(0);   // width
    auto c  = input->dimension(1);   // height
    auto m  = weights->dimension(1); // height
    auto kk = weights->dimension(2); // depth

    ARM_COMPUTE_ERROR_ON(c != weights->dimension(0));

    return Status{};
}

void AccumulatingGEMM::run()
{
    ARM_COMPUTE_ERROR("AccumulatingGEMM::run() not supported. Use AccumulatingGEMM::run(size_t filter_perforation) instead.");
}

void AccumulatingGEMM::run(size_t filter_perforation)
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    for(size_t i = 0; i < KK; i++)
    {
        if(is_central_element_index(KK, i) || filter_perforation == 1 || i % filter_perforation == 0)
        {
            CLScheduler::get().enqueue(*_filter_image_gemmkernels[i]);
            CLScheduler::get().sync();

            // TODO: accumulate results
        }
    }
}

void AccumulatingGEMM::prepare()
{
}
