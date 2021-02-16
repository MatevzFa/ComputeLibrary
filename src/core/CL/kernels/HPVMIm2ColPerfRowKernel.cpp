/*
 * Copyright (c) 2017-2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "src/core/CL/kernels/HPVMIm2ColPerfRowKernel.h"

#include "CL/cl_platform.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <cmath>
#include <tuple>
#include <utility>

#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "ARMComputeLibrary", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ARMComputeLibrary", __VA_ARGS__)

namespace arm_compute
{
using namespace misc::shape_calculator;

namespace
{
inline TensorShape compute_hpvm_im2col_perfrow_conv_shape(const ITensorInfo *input, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const HPVMIm2ColPerfInfo &perf_info, const Size2D &dilation, bool batch_size_on_z,
                                                          unsigned int num_groups = 1)
{
    // The output shape will be the 3D shape [ out_channels * kernel_area, num_elems_per_out_channel, batches ]                           if batch_size_on_z == true
    //                       or the 4D shape [ out_channels * kernel_area / num_groups, num_elems_per_out_channel, num_groups, batches ]  if batch_size_on_z == false

    ARM_COMPUTE_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_ERROR_ON(num_groups > 1 && input->data_layout() != DataLayout::NCHW);
    ARM_COMPUTE_ERROR_ON(num_groups > 1 && batch_size_on_z);

    TensorShape output_shape{ input->tensor_shape() };

    const DataLayout data_layout = input->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    std::pair<unsigned int, unsigned int> out_dims = scaled_dimensions(output_shape[width_idx], output_shape[height_idx], kernel_dims.width, kernel_dims.height, conv_info, dilation);
    // Skip every -th filter element
    output_shape.set(0, (output_shape[channel_idx] / num_groups * (kernel_dims.area() - (kernel_dims.area() / perf_info.perffilter_every)) + (has_bias ? 1 : 0))); // NOLINT
    // Skip every row
    output_shape.set(1, (out_dims.first * (out_dims.second / perf_info.perfrow_every)));
    if(batch_size_on_z && output_shape.num_dimensions() >= 3)
    {
        output_shape.remove_dimension(2);
    }
    else
    {
        output_shape.set(2, num_groups);
    }

    return output_shape;
}

struct Im2ColConfiguration
{
    std::string           kernel_name{};
    std::set<std::string> build_options{};
    unsigned int          num_elems_processed_per_iteration{};
    bool                  is_padding_required_nchw{};
};

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const HPVMIm2ColPerfInfo &perf_info, const Size2D &dilation,
                          unsigned int num_groups)
{
    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized(input->data_type()) && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::NHWC && num_groups > 1);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(channel_idx) % num_groups) != 0);

    // Since there's no implicit padding added, check the total input spatial dimensions (with conv paddings) are big enough for the kernel dimensions
    const unsigned int width_idx    = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int height_idx   = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const unsigned     total_width  = input->dimension(width_idx) + conv_info.pad_left() + conv_info.pad_right();
    const unsigned     total_height = input->dimension(height_idx) + conv_info.pad_top() + conv_info.pad_bottom();
    ARM_COMPUTE_RETURN_ERROR_ON((total_width < kernel_dims.width) || (total_height < kernel_dims.height));

    if(output->total_size() > 0)
    {
        const TensorInfo tensor_info_output = output->clone()->set_tensor_shape(compute_hpvm_im2col_perfrow_conv_shape(input, kernel_dims, conv_info, has_bias, perf_info, dilation, num_groups == 1, num_groups));
        LOGE("expected %ld %ld %ld %ld",
             tensor_info_output.tensor_shape()[0],
             tensor_info_output.tensor_shape()[1],
             tensor_info_output.tensor_shape()[2],
             tensor_info_output.tensor_shape()[3]);
        LOGE("got %ld %ld %ld %ld",
             output->tensor_shape()[0],
             output->tensor_shape()[1],
             output->tensor_shape()[2],
             output->tensor_shape()[3]);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const HPVMIm2ColPerfInfo &perf_info, const Size2D &dilation,
                                                        unsigned int num_elems_processed_per_iteration, bool is_padding_required_nchw, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto initialization if not yet initialized
    TensorShape expected_output_shape = compute_hpvm_im2col_perfrow_conv_shape(input, kernel_dims, conv_info, has_bias, perf_info, dilation, num_groups == 1, num_groups);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(expected_output_shape));

    const DataLayout   data_layout  = input->data_layout();
    const unsigned int width_idx    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int input_width  = input->dimension(width_idx);
    const unsigned int input_height = input->dimension(height_idx);

    // Configure the execute window based on the selected optimal OpenCL kernel
    bool   window_changed = false;
    Window win;

    if(data_layout == DataLayout::NHWC)
    {
        win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

        const int xin_start = 0;
        const int xin_end   = input->dimension(0);
        const int yin_start = 0;
        const int yin_end   = input->dimension(1);

        const int xout_start = 0;
        const int xout_end   = output->dimension(0);
        const int yout_start = 0;
        const int yout_end   = output->dimension(1);

        AccessWindowStatic input_access(input, xin_start, yin_start, xin_end, yin_end);
        AccessWindowStatic output_access(output, xout_start, yout_start, xout_end, yout_end);
        window_changed = window_changed || update_window_and_padding(win, input_access, output_access);
    }
    else
    {
        if(is_padding_required_nchw)
        {
            const BorderSize border(conv_info.pad_top(), conv_info.pad_right(), conv_info.pad_bottom(), conv_info.pad_left());
            win = calculate_max_window(*input,
                                       Steps(num_elems_processed_per_iteration * conv_info.stride().first, conv_info.stride().second));
            AccessWindowStatic input_access(input,
                                            -border.left,
                                            -border.top,
                                            ceil_to_multiple(input_width + border.right, kernel_dims.width * num_elems_processed_per_iteration),
                                            input_height + border.bottom);
            window_changed = window_changed || update_window_and_padding(win, input_access);
        }
        else
        {
            // For the generic case, HPVMIm2ColPerfRowKernel doesn't need padding (we do not read out-of-bounds elements) so
            // update_window_and_padding() can be skipped
            win = calculate_max_window(*input, Steps());
        }
    }
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
    // set the Z dimension's step same size as the whole dimension so that one can't split across the Z dimension
    win.set_dimension_step(Window::DimZ, win[Window::DimZ].end() - win[Window::DimZ].start());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Im2ColConfiguration configure_opencl_kernel(const ITensorInfo *input, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    const DataLayout   data_layout   = input->data_layout();
    const DataType     data_type     = input->data_type();
    const unsigned int width_idx     = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const unsigned int input_width   = input->dimension(width_idx);
    const unsigned int input_height  = input->dimension(height_idx);
    const unsigned int input_channel = input->dimension(channel_idx);

    const std::pair<unsigned int, unsigned int> convolved_dims = scaled_dimensions(input_width, input_height, kernel_dims.width, kernel_dims.height, conv_info, dilation);

    LOGE("convolved_dims w=%d h=%d", convolved_dims.first, convolved_dims.second);

    // Im2Col configuration
    std::string                   kernel_name = "hpvm_im2col_perfrow_generic_nchw";
    CLBuildOptions                build_opts;
    unsigned int                  num_elems_processed_per_iteration = 1;
    bool                          is_padding_required_nchw          = false;
    const UniformQuantizationInfo qinfo                             = input->quantization_info().uniform();

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input->element_size()));
    build_opts.add_option("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
    build_opts.add_option("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
    build_opts.add_option("-DCONVOLVED_WIDTH=" + support::cpp11::to_string(convolved_dims.first));
    build_opts.add_option("-DCONVOLVED_HEIGHT=" + support::cpp11::to_string(convolved_dims.second));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
    build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.add_option("-DPAD_RIGHT=" + support::cpp11::to_string(conv_info.pad_right()));
    build_opts.add_option("-DPAD_BOTTOM=" + support::cpp11::to_string(conv_info.pad_bottom()));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input_width));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input_height));
    build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input_channel));
    build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
    build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));
    build_opts.add_option_if(num_groups > 1, "-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));
    build_opts.add_option_if_else(is_data_type_quantized(data_type), "-DPAD_VALUE=" + support::cpp11::to_string(qinfo.offset), "-DPAD_VALUE=0");
    build_opts.add_option_if(has_bias, "-DHAS_BIAS");

    if(data_layout != DataLayout::NCHW)
    {
        ARM_COMPUTE_ERROR("HPVMIm2ColPerfRowKernel: Unsupported configuration");
    }

    Im2ColConfiguration im2col_config;
    im2col_config.kernel_name                       = kernel_name;
    im2col_config.build_options                     = build_opts.options();
    im2col_config.num_elems_processed_per_iteration = num_elems_processed_per_iteration;
    im2col_config.is_padding_required_nchw          = is_padding_required_nchw;

    return im2col_config;
}
} // namespace

HPVMIm2ColPerfRowKernel::HPVMIm2ColPerfRowKernel()
    : _input(nullptr), _output(nullptr), _data_layout(DataLayout::UNKNOWN), _convolved_dims(), _num_elems_processed_per_iteration(1), _kernel_dims(), _conv_info(), _num_groups()
{
}

void HPVMIm2ColPerfRowKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias,
                                        const HPVMIm2ColPerfInfo &perf_info,
                                        const Size2D &dilation, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(),
              input, output, kernel_dims, conv_info, has_bias,
              perf_info,
              dilation, num_groups);
}

void HPVMIm2ColPerfRowKernel::configure(const CLCompileContext &compile_context,
                                        const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias,
                                        const HPVMIm2ColPerfInfo &perf_info,
                                        const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, perf_info, dilation, num_groups));

    auto padding_info = get_padding_info({ input, output });
    _data_layout      = input->info()->data_layout();

    _perf_info = perf_info;

    const unsigned int width_idx    = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int input_width  = input->info()->dimension(width_idx);
    const unsigned int input_height = input->info()->dimension(height_idx);

    // Select and configure the optimal OpenCL kernel to run.
    // This function returns the OpenCL kernel's name, the arguments to pass at compile time, the number of elements processed per iteration
    // and the padding requirement flag
    Im2ColConfiguration im2col_config = configure_opencl_kernel(input->info(), kernel_dims, conv_info, has_bias, dilation, num_groups);

    // Create kernel
    _kernel = create_kernel(compile_context, im2col_config.kernel_name, im2col_config.build_options);

    _input                             = input;
    _output                            = output;
    _convolved_dims                    = scaled_dimensions(input_width, input_height, kernel_dims.width, kernel_dims.height, conv_info, dilation);
    _num_elems_processed_per_iteration = im2col_config.num_elems_processed_per_iteration;
    _kernel_dims                       = kernel_dims; // Only needed by the Tuner
    _conv_info                         = conv_info;   // Only needed by the Tuner
    _num_groups                        = num_groups;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), kernel_dims, conv_info, has_bias, perf_info, dilation, im2col_config.num_elems_processed_per_iteration,
                                                    im2col_config.is_padding_required_nchw, num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = im2col_config.kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(num_groups);
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(_data_layout));

    ARM_COMPUTE_ERROR_ON(input->info()->data_layout() == DataLayout::NHWC && has_padding_changed(padding_info));
}

Status HPVMIm2ColPerfRowKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const HPVMIm2ColPerfInfo &perf_info, const Size2D &dilation,
                                         unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, perf_info, dilation, num_groups));
    Im2ColConfiguration im2col_config = configure_opencl_kernel(input, kernel_dims, conv_info, has_bias, dilation, num_groups);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), kernel_dims, conv_info, has_bias, perf_info, dilation, im2col_config.num_elems_processed_per_iteration,
                                                              im2col_config.is_padding_required_nchw, num_groups)
                                    .first);
    return Status{};
}

void HPVMIm2ColPerfRowKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    // Get initial windows
    // Collapse in order to have (SRC_DEPTH * BATCH_SIZE) on the 3rd dimension
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    window_collapsed.set_dimension_step(Window::DimZ, 1);

    Window window_output;
    window_output.use_tensor_dimensions(_output->info()->tensor_shape());

    const Window first_slice_3d = window_collapsed.first_slice_window_3D();

    Window slice     = first_slice_3d;
    Window slice_in  = first_slice_3d;
    Window slice_out = window_output.first_slice_window_2D();

    if(_data_layout == DataLayout::NHWC)
    {
        const Window tmp_win     = window.collapse_if_possible(ICLKernel::window(), 3);
        const int    num_batches = tmp_win[3].end();

        slice.set(1, Window::Dimension(0, static_cast<int>(_output->info()->tensor_shape()[1]), 1));
        slice.set(2, Window::Dimension(0, static_cast<int>(num_batches), 1));
    }
    else
    {
        slice.set(0, Window::Dimension(0, static_cast<int>(ceil_to_multiple(_convolved_dims.first, _num_elems_processed_per_iteration)), _num_elems_processed_per_iteration));
        slice.set(1, Window::Dimension(0, static_cast<int>(_convolved_dims.second), 1));
        // Note: In case of NCHW the 3rd dimension is already set collapsing the input window
    }

    // Setup input slice
    // The dimensions of the input are increased within the OpenCL kernel
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Setup output slice
    // The dimensions of the output are increased within the OpenCL kernel
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    unsigned int idx = num_arguments_per_3D_tensor() + (_num_groups == 1 ? num_arguments_per_2D_tensor() : num_arguments_per_3D_tensor());
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input->info()->strides_in_bytes()[3]));
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[((_num_groups == 1) ? 2 : 3)]));
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_perf_info.perfrow_start));
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_perf_info.perfrow_every));
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_perf_info.perffilter_start));
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_perf_info.perffilter_every));
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        if(_num_groups == 1)
        {
            add_2D_tensor_argument(idx, _output, slice_out);
        }
        else
        {
            add_3D_tensor_argument(idx, _output, slice_out);
        }
        enqueue(queue, *this, slice, lws_hint());
    } while(window_collapsed.slide_window_slice_3D(slice) && window_output.slide_window_slice_2D(slice_out) && window_collapsed.slide_window_slice_3D(slice_in));
}
} // namespace arm_compute
