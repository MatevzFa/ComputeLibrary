#include "src/core/CL/kernels/HPVMInterpolateKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/functions/HPVMConvApprox.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
HPVMInterpolateKernel::HPVMInterpolateKernel(){};

void HPVMInterpolateKernel::configure(const ICLTensor *input, ICLTensor *output,
                                      const HPVMConvApproxInfo &perf_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, perf_info);
}

void HPVMInterpolateKernel::configure(const CLCompileContext &compile_context,
                                      const ICLTensor *input, ICLTensor *output,
                                      const HPVMConvApproxInfo &perf_info)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), perf_info));

    _input     = input;
    _output    = output;
    _perf_info = perf_info;

    auto n_idx = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::BATCHES);
    auto c_idx = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);
    auto h_idx = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    auto w_idx = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);

    auto in_n = input->info()->dimension(n_idx);
    auto in_c = input->info()->dimension(c_idx);
    auto in_h = input->info()->dimension(h_idx);
    auto in_w = input->info()->dimension(w_idx);

    auto out_n = output->info()->dimension(n_idx);
    auto out_c = output->info()->dimension(c_idx);
    auto out_h = output->info()->dimension(h_idx);
    auto out_w = output->info()->dimension(w_idx);

    CLBuildOptions opts{};
    opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    if(perf_info.mode == HPVMConvApproxPerfMode::ROW)
    {
        _kernel = create_kernel(compile_context, "hpvm_interpolate_row", opts.options());

        int idx = 2 * num_arguments_per_4D_tensor();

        _kernel.setArg<cl_uint>(idx++, static_cast<cl_uint>(out_h));
        _kernel.setArg<cl_uint>(idx++, perf_info.perf_start);
        _kernel.setArg<cl_uint>(idx++, perf_info.perf_every);

        Window win = calculate_max_window(*output->info());

        configure_internal(win);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported");
    }
}

Status HPVMInterpolateKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const HPVMConvApproxInfo &perf_info)
{
    ARM_COMPUTE_UNUSED(input, output, perf_info);

    return Status{};
}

void HPVMInterpolateKernel::run(const Window &window, cl::CommandQueue &queue)
{
    auto slice = window.first_slice_window_4D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    } while(slice.slide_window_slice_4D(slice));
}

}; // namespace arm_compute
