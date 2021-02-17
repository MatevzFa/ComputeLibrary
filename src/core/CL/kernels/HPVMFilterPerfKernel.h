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
#ifndef ARM_COMPUTE_HPVMFILTERPERFROWKERNEL_H
#define ARM_COMPUTE_HPVMFILTERPERFROWKERNEL_H

#include "arm_compute/core/Size2D.h"

#include "src/core/CL/ICLKernel.h"

#include <cstddef>

namespace arm_compute
{
class ICLTensor;

struct HPVMFilterPerfInfo
{
    size_t perf_start = 0;
    size_t perf_every = 0;

    HPVMFilterPerfInfo()
    {
    }

    HPVMFilterPerfInfo(size_t start, size_t every)
    {
        perf_start = start;
        perf_every = every;
    };
};

/** Interface for the im2col reshape kernel.
 *
 * Rearranges image blocks into columns. It is used to strip out each convolution block to a single column.
 * It is used to transform a convolution to a plain matrix multiplication.
 *
 * For example taking into account the image below and assuming 3x3 image blocks with stride of 1 we have:
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * =
 * \left( \begin{array}{ccccccccc}
 * a00 & a01 & a02 & a10 & a11 & a12 & a20 & a21 & a22 \\
 * a01 & a02 & a03 & a11 & a12 & a13 & a21 & a22 & a23 \\
 * a10 & a11 & a12 & a20 & a21 & a22 & a30 & a31 & a32 \\
 * a11 & a12 & a13 & a21 & a22 & a23 & a31 & a32 & a33 \\
 * \end{array} \right)
 * @f]
 */
class HPVMFilterPerfKernel : public ICLKernel
{
public:
    void configure(const ICLTensor *input, ICLTensor *output,
                   const HPVMFilterPerfInfo &perf_info);

    void configure(const CLCompileContext &compile_context,
                   const ICLTensor *input, ICLTensor *output,
                   const HPVMFilterPerfInfo &perf_info);

    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const HPVMFilterPerfInfo &perf_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    const ICLTensor *  _input;
    ICLTensor *        _output;
    HPVMFilterPerfInfo _perf_info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_HPVMFILTERPERFROWKERNEL_H */
