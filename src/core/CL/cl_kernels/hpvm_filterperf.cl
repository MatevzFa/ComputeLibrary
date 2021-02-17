/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "helpers.h"

__kernel void hpvm_filterperf(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_num_filter_elements,
    uint dst_num_filter_elements,
    uint channels,
    uint batches,
    uint perffilter_start,
    uint perffilter_every)
{
    int x = get_global_id(0); // [0, 2)
    int y = get_global_id(1); // [0, 15)

    int chan  = y / dst_num_filter_elements;
    int batch = x;

    int elem_in    = y + perffilter_start + y / (perffilter_every - 1);
    int yin_offset = src_num_filter_elements * channels * batch + src_num_filter_elements * chan + elem_in;

    __global DATA_TYPE *input_ptr  = (__global DATA_TYPE *)(src_ptr + src_offset_first_element_in_bytes) + yin_offset;
    __global DATA_TYPE *output_ptr = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes) + y * batches + batch;

    *output_ptr = *input_ptr;
}
