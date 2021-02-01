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

#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/ICLSimple2DKernel.h"
#include "src/core/CL/ICLSimpleKernel.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Interface for the accumulate kernel.
 *
 * Accumulation is computed by:
 * @f[ accum(x,y) = accum(x,y) + input(x+offset_x,y+offset_y) @f]
 */
class HPVMAccumulateKernel : public ICLSimple2DKernel
{
public:
    /** Set the input and accumulation tensors.
     *
     * @param[in]  input Source tensor. Data types supported: U8.
     * @param[out] accum Destination tensor. Data types supported: S16.
     */
    void configure(ICLTensor *accum, const ICLTensor *input,
                   const size_t w, const size_t h, const size_t m,
                   const size_t offset_w, const size_t offset_h);
    /** Set the input and accumulation tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: U8.
     * @param[out] accum           Destination tensor. Data types supported: S16.
     */
    void configure(const CLCompileContext &compile_context,
                   ICLTensor *accum, const ICLTensor *input,
                   const size_t w, const size_t h, const size_t m,
                   const size_t offset_w, const size_t offset_h);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLACCUMULATEKERNEL_H */
