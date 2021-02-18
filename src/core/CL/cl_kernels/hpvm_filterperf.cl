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
    uint kw, uint kh,
    uint channels,
    uint batches,
    uint perffilter_start,
    uint perffilter_every)
{
    int x = get_global_id(0); // [0, nfilters)
    int y = get_global_id(1); // [0, dst_num_filter_elements)

    int chan  = y / dst_num_filter_elements;
    int batch = x;

    int elem_in     = y + perffilter_start + y / (perffilter_every - 1);
    int elem_in_row = elem_in / kw;
    int elem_in_col = elem_in % kw;

    int in_batch_offset   = batch * channels * kw * src_stride_y;
    int in_channel_offset = chan * kw * src_stride_y;
    int in_row_offset     = elem_in_row * src_stride_y;
    int in_col_offset     = (elem_in_col);

    int out_row_offset = y * dst_stride_y;
    int out_col_offset = x;

    __global DATA_TYPE *input_ptr  = (__global DATA_TYPE *)(src_ptr + src_offset_first_element_in_bytes + in_batch_offset + in_channel_offset + in_row_offset) + in_col_offset;
    __global DATA_TYPE *output_ptr = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + out_row_offset) + out_col_offset;

    DATA_TYPE fac = 1;
    if(perffilter_every >= 2)
    {
        // fac = (DATA_TYPE)src_num_filter_elements / (DATA_TYPE)dst_num_filter_elements;
        fac = (DATA_TYPE)perffilter_every / ((DATA_TYPE)perffilter_every - 1);
    }
    *output_ptr = *input_ptr * fac;
}
