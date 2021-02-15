/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLACCUMULATEKERNEL_H
#define ARM_COMPUTE_CLACCUMULATEKERNEL_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/ICLSimple2DKernel.h"
#include "src/core/CL/ICLSimpleKernel.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Interface for the accumulate kernel.
 *
 * Accumulation is computed by:
 * @f[ accum(x,y) = accum(x,y) + input(x+offset_x,y+offset_y) @f]
 */
class HPVMIm2ColPerfRowKernel : public ICLKernel
{
public:
    /** Set the input and accumulation tensors.
     *
     * @param[in]  input Source tensor. Data types supported: U8.
     * @param[out] accum Destination tensor. Data types supported: S16.
     */
    void configure(const ICLTensor *input, ICLTensor *output,
                   const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                   size_t perfrow_start, size_t perfrow_every, size_t perfrow_h_eff)
    {
        configure(CLKernelLibrary::get().get_compile_context(), input, output, kernel_dims, conv_info, perfrow_start, perfrow_every, perfrow_h_eff);
    }

    /** Set the input and accumulation tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: U8.
     * @param[out] accum           Destination tensor. Data types supported: S16.
     */
    void configure(const CLCompileContext &compile_context,
                   const ICLTensor *input, ICLTensor *output,
                   const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                   size_t perfrow_start, size_t perfrow_every, size_t perfrow_h_eff)
    {
        _input  = input;
        _output = output;

        auto input_info  = input->info();
        auto output_info = output->info();
        auto data_type   = input_info->data_type();

        _n = input_info->dimension(DataLayoutDimension::BATCHES);
        _c = input_info->dimension(DataLayoutDimension::CHANNEL);
        _h = input_info->dimension(DataLayoutDimension::HEIGHT);
        _w = input_info->dimension(DataLayoutDimension::WIDTH);

        _kh = kernel_dims.height;
        _kw = kernel_dims.width;

        _perfrow_start = perfrow_start;
        _perfrow_every = perfrow_every;
        _perfrow_h_eff = perfrow_h_eff;

        auto stride_x = conv_info.stride().first;
        auto stride_y = conv_info.stride().second;

        size_t out_w = (_w + stride_x - 1) / stride_x;
        size_t out_h = (_h + stride_y - 1) / stride_y;
        out_h        = (out_h + perfrow_every - 1) / perfrow_every;

        size_t convolved_height = _kh * _kw;
        size_t convolved_width  = _n * _c * out_h * out_w;

        ARM_COMPUTE_ERROR_ON(output_info->dimension(DataLayoutDimension::HEIGHT) != convolved_height);
        ARM_COMPUTE_ERROR_ON(output_info->dimension(DataLayoutDimension::WIDTH) != convolved_width);

        CLBuildOptions opts{};
        opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
        opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input_info->element_size()));
        opts.add_option("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
        opts.add_option("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
        opts.add_option("-DCONVOLVED_HEIGHT=" + support::cpp11::to_string(convolved_height));
        opts.add_option("-DCONVOLVED_WIDTH=" + support::cpp11::to_string(convolved_width));
        opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
        opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
        opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
        opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
        opts.add_option("-DPAD_RIGHT=" + support::cpp11::to_string(conv_info.pad_right()));
        opts.add_option("-DPAD_BOTTOM=" + support::cpp11::to_string(conv_info.pad_bottom()));
        opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(_w));
        opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(_h));
        opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(_c));
        opts.add_option("-DPAD_VALUE=0");

        _kernel = create_kernel(compile_context, "hpvm_im2col_perfrow_generic_nchw", opts.options());

        unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
        _kernel.setArg(idx++, (unsigned int)input_info->strides_in_bytes()[3]);
        _kernel.setArg(idx++, (unsigned int)output_info->strides_in_bytes()[3]);
        _kernel.setArg(idx++, (unsigned int)_perfrow_start);
        _kernel.setArg(idx++, (unsigned int)_perfrow_every);
        _kernel.setArg(idx++, (unsigned int)_perfrow_h_eff);

        Window win = calculate_max_window(*output_info);

        // TODO: tune local size
        ICLKernel::configure_internal(win);
        // ICLKernel::configure_internal(win, cl::NDRange(16u, 1u));

        std::ostringstream config_id_str;
        config_id_str << "hpvm_im2col_perfrow_generic_nchw" << string_from_data_type(data_type) << "_"
                      << _w << "_" << _h << "_" << _c << "_" << _n;

        _config_id = config_id_str.str();
    }

    void run(const Window &window, cl::CommandQueue &queue) override
    {
        ITensorPack pack{};
        pack.add_tensor(TensorType::ACL_SRC, _input);
        pack.add_tensor(TensorType::ACL_DST, _output);
        run_op(pack, window, queue);
    }

    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override

    {
        Window slice = window.first_slice_window_2D();

        const auto input  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
        auto       output = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

        // Set inputs
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, input, slice);
        add_2D_tensor_argument(idx, output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }

private:
    const ICLTensor *_input;
    const ICLTensor *_output;

    size_t _n, _c, _h, _w;
    size_t _kh, _kw;
    size_t _perfrow_start, _perfrow_every, _perfrow_h_eff;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLACCUMULATEKERNEL_H */
