#include "src/core/CL/kernels/HPVMFilterPerfKernel.h"
#include "CL/cl_platform.h"
#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/WindowHelpers.h"

using namespace arm_compute;

void HPVMFilterPerfKernel::configure(const ICLTensor *input, ICLTensor *output,
                                     const HPVMFilterPerfInfo &perf_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, perf_info);
}

void HPVMFilterPerfKernel::configure(const CLCompileContext &compile_context,
                                     const ICLTensor *input, ICLTensor *output,
                                     const HPVMFilterPerfInfo &perf_info)
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

    auto in_filter_elements  = in_h * in_w;
    auto out_filter_elements = in_filter_elements - (perf_info.perf_every < 2 ? 0 : in_filter_elements / perf_info.perf_every);

    CLBuildOptions opts{};
    opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    _kernel = create_kernel(compile_context, "hpvm_filterperf", opts.options());

    int idx = num_arguments_per_4D_tensor() + num_arguments_per_2D_tensor();
    _kernel.setArg<cl_uint>(idx++, in_filter_elements);
    _kernel.setArg<cl_uint>(idx++, out_filter_elements);
    _kernel.setArg<cl_uint>(idx++, in_w);
    _kernel.setArg<cl_uint>(idx++, in_h);
    _kernel.setArg<cl_uint>(idx++, in_c);
    _kernel.setArg<cl_uint>(idx++, in_n);
    _kernel.setArg<cl_uint>(idx++, perf_info.perf_start);
    _kernel.setArg<cl_uint>(idx++, perf_info.perf_every);

    Window win = calculate_max_window(*output->info());

    configure_internal(win);
}

Status HPVMFilterPerfKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const HPVMFilterPerfInfo &perf_info)
{
    ARM_COMPUTE_UNUSED(input, output, perf_info);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(perf_info.perf_start != 0, "Only perf_start == 0 supported");
    // ARM_COMPUTE_RETURN_ERROR_ON(perf_info.perf_every < 2);

    auto n_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::BATCHES);
    auto c_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
    auto h_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    auto w_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);

    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != output->data_layout());

    auto in_n = input->dimension(n_idx);
    auto in_c = input->dimension(c_idx);
    auto in_h = input->dimension(h_idx);
    auto in_w = input->dimension(w_idx);

    auto out_n = output->dimension(n_idx);
    auto out_c = output->dimension(c_idx);
    auto out_h = output->dimension(h_idx);
    auto out_w = output->dimension(w_idx);

    auto in_filter_elements  = in_h * in_w;
    auto out_filter_elements = in_filter_elements - (perf_info.perf_every < 2 ? 0 : in_filter_elements / perf_info.perf_every);

    ARM_COMPUTE_RETURN_ERROR_ON(out_n != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(out_c != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(out_h != in_c * out_filter_elements);
    ARM_COMPUTE_RETURN_ERROR_ON(out_w != in_n);

    return Status{};
}

void HPVMFilterPerfKernel::run(const Window &window, cl::CommandQueue &queue)
{
    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    } while(window.slide_window_slice_2D(slice));
}
