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
#include "src/core/CL/kernels/HPVMAccumulateKernel.h"

#include "CL/cl2.hpp"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/ICLSimpleKernel.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include <sstream>
#include <string>

namespace arm_compute
{
void HPVMAccumulateKernel::configure(ICLTensor *accum, const ICLTensor *input,
                                     const size_t w, const size_t h, const size_t m,
                                     const long offset_w, const long offset_h)
{
    configure(CLKernelLibrary::get().get_compile_context(), accum, input, w, h, m, offset_w, offset_h);
}

void HPVMAccumulateKernel::configure(const CLCompileContext &compile_context, ICLTensor *accum, const ICLTensor *input,
                                     const size_t w, const size_t h, const size_t m,
                                     const long offset_w, const long offset_h)
{
    _accum = accum;
    _input = input;

    const std::set<std::string> build_opts{
        "-DDATA_TYPE=" + get_cl_type_from_data_type(accum->info()->data_type())
    };

    // Create kernel
    _kernel = create_kernel(compile_context, "hpvm_add_offset", build_opts);

    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the accum and input parameters
    _kernel.setArg(idx++, w);
    _kernel.setArg(idx++, h);
    _kernel.setArg(idx++, h * w);
    _kernel.setArg(idx++, m);
    _kernel.setArg(idx++, offset_w);
    _kernel.setArg(idx++, offset_h);

    Window win = calculate_max_window(*accum->info());

    ICLKernel::configure_internal(win, cl::NDRange(16u, 1u));

    std::ostringstream config_id_str;
    config_id_str << "hpvm_add_offset_" << string_from_data_type(accum->info()->data_type()) << "_"
                  << w << "_" << h << "_" << m << "_"
                  << offset_w << "_" << offset_h;

    _config_id += config_id_str.str();
}

void HPVMAccumulateKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ITensorPack pack{};
    pack.add_tensor(TensorType::ACL_SRC, _input);
    pack.add_tensor(TensorType::ACL_DST, _accum);
    run_op(pack, window, queue);
}

void HPVMAccumulateKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    Window slice = window.first_slice_window_2D();

    const auto input = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       accum = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Set inputs
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, input, slice);
    add_2D_tensor_argument(idx, accum, slice);
    enqueue(queue, *this, slice, lws_hint());
}
} // namespace arm_compute
