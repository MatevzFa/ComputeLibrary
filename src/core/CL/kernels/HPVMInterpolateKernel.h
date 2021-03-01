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
#ifndef ARM_COMPUTE_HPVMINTERPOLATEKERNEL_H
#define ARM_COMPUTE_HPVMINTERPOLATEKERNEL_H

#include "arm_compute/core/Size2D.h"
#include "arm_compute/runtime/HPVMTypes.h"

#include "src/core/CL/ICLKernel.h"

#include <cstddef>

namespace arm_compute
{
class ICLTensor;

class HPVMInterpolateKernel : public ICLKernel
{
public:
    /** Default constructor */
    HPVMInterpolateKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    HPVMInterpolateKernel(const HPVMInterpolateKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    HPVMInterpolateKernel &operator=(const HPVMInterpolateKernel &) = delete;
    /** Allow instances of this class to be moved */
    HPVMInterpolateKernel(HPVMInterpolateKernel &&) = default;
    /** Allow instances of this class to be moved */
    HPVMInterpolateKernel &operator=(HPVMInterpolateKernel &&) = default;

    void configure(const ICLTensor *input, ICLTensor *output,
                   const HPVMConvApproxInfo &perf_info);

    void configure(const CLCompileContext &compile_context,
                   const ICLTensor *input, ICLTensor *output,
                   const HPVMConvApproxInfo &perf_info);

    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const HPVMConvApproxInfo &perf_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    const ICLTensor *  _input;
    ICLTensor *        _output;
    HPVMConvApproxInfo _perf_info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_HPVMINTERPOLATEKERNEL_H */
