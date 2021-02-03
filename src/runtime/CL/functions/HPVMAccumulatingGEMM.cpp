
#include "arm_compute/runtime/CL/functions/HPVMAccumulatingGEMM.h"

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
#include "arm_compute/runtime/CL/functions/CLFill.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/SubTensor.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyNativeKernel.h"
#include "support/MemorySupport.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "android/log.h"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "AccumulatingGEMM", __VA_ARGS__)

using namespace arm_compute;

void log_dims(const char *name, ITensorInfo *info)
{
    LOGI("[%s %ld %ld %ld %ld]",
         name,
         info->dimension(0), info->dimension(1), info->dimension(2), info->dimension(3));
}

std::string tensor_to_string(ICLTensor *tensor)
{
    tensor->map(CLScheduler::get().queue());

    std::ostringstream stream;
    tensor->print(stream);

    tensor->unmap(CLScheduler::get().queue());

    return stream.str();
}

size_t central_index(size_t k)
{
    return k / 2 * (k + 1);
}

std::pair<size_t, size_t> kernel_coords(size_t k, size_t kernel_index)
{
    int row = kernel_index / k;
    int col = kernel_index % k;

    return std::make_pair(row, col);
}

std::pair<long, long> kernel_offset(std::pair<size_t, size_t> central_coords, std::pair<size_t, size_t> coords)
{
    long i_row, i_col;
    std::tie(i_row, i_col) = coords;

    long c_row, c_col;
    std::tie(c_row, c_col) = central_coords;

    return std::make_pair(i_row - c_row, i_col - c_col);
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
                                 const size_t w, const size_t h,
                                 const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                 const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(),
              input, weights, output,
              w, h,
              conv_info, weights_info,
              dilation, act_info, num_groups);
}

void AccumulatingGEMM::configure(const CLCompileContext &compile_context,
                                 const ICLTensor *input, ICLTensor *weights, ICLTensor *output,
                                 const size_t w, const size_t h,
                                 const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                 const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), weights->info(), output->info(), w, h, conv_info, weights_info, dilation, act_info, num_groups));

    C  = weights->info()->dimension(0);
    M  = weights->info()->dimension(1);
    KK = weights->info()->dimension(2);
    W  = w;
    H  = h;

    LOGI("C %ld", C);
    LOGI("M %ld", M);
    LOGI("KK %ld", KK);
    LOGI("HW %ld", H * W);

    log_dims("input", input->info());
    log_dims("weights", weights->info());

    _output_tensor_ptr = output;
    _output_tensor_aux.allocator()->init(*_output_tensor_ptr->info());

    log_dims("_output_buffer", _output_tensor_aux.info());
    log_dims("_output_tensor", _output_tensor_ptr->info());

    _memory_group.manage(&_output_tensor_aux);

    _fill_func.configure(&_output_tensor_aux, 0);

    GEMMLHSMatrixInfo lhs_info{};
    lhs_info.m0 = 1;
    lhs_info.k0 = 2;

    GEMMRHSMatrixInfo rhs_info{};
    rhs_info.n0 = 2;
    rhs_info.k0 = lhs_info.k0;

    int  _k             = sqrt(KK);
    auto central_coords = kernel_coords(_k, central_index(_k));

    for(size_t i = 0; i < KK; i++)
    {
        auto filters_view = support::cpp14::make_unique<CLSubTensor>(weights, TensorShape(C, M, 1), Coordinates(0, 0, i));
        _subtensors.push_back(std::move(filters_view));

        GEMMInfo gemm_info{};

        auto gemm = support::cpp14::make_unique<CLGEMM>();
        gemm->configure(compile_context, _subtensors[i].get(), input, nullptr, &_output_tensor_aux, 1, 0, gemm_info);
        _filter_image_mm.push_back(std::move(gemm));

        long offset_w, offset_h;
        std::tie(offset_w, offset_h) = kernel_offset(central_coords, kernel_coords(_k, i));

        auto k_accum = support::cpp14::make_unique<HPVMAccumulateKernel>();
        k_accum->configure(compile_context, _output_tensor_ptr, &_output_tensor_aux, W, H, M, offset_w, offset_h);
        _output_accum_kernels.push_back(std::move(k_accum));
    }

    _output_tensor_aux.allocator()->allocate();
}

Status AccumulatingGEMM::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output,
                                  const size_t w, const size_t h,
                                  const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                  const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    auto hw = input->dimension(0);   // width
    auto c  = input->dimension(1);   // height
    auto m  = weights->dimension(1); // height
    auto kk = weights->dimension(2); // depth

    ARM_COMPUTE_ERROR_ON(h * w != hw);

    ARM_COMPUTE_ERROR_ON(c != weights->dimension(0));

    ARM_COMPUTE_ERROR_ON(hw != output->dimension(0));
    ARM_COMPUTE_ERROR_ON(m != output->dimension(1));

    return Status{};
}

void AccumulatingGEMM::run()
{
    ARM_COMPUTE_ERROR("AccumulatingGEMM::run() not supported. Use AccumulatingGEMM::run(size_t filter_perforation) instead.");
}

void AccumulatingGEMM::run(size_t skip_every)
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    for(size_t i = 0; i < KK; i++)
    {
        if(skip_every == 0 || (i + 1) % skip_every != 0)
        {
            _filter_image_mm[i]->run();
            // CLScheduler::get().sync();

            CLScheduler::get().enqueue(*_output_accum_kernels[i]);
            // CLScheduler::get().sync();

            // LOGI(
            //     "at %ld\n"
            //     "_output_tensor_aux: %s\n"
            //     "_output_tensor_ptr:  %s",
            //     i, tensor_to_string(&_output_tensor_aux).c_str(), tensor_to_string(_output_tensor_ptr).c_str());
        }
    }
}

void AccumulatingGEMM::prepare()
{
    _fill_func.run();
    CLScheduler::get().sync();
}

bool AccumulatingGEMM::is_central_element_index(size_t index)
{
    size_t _K = sqrt(KK);
    return index == central_index(_K);
}
