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

#if defined(DATA_TYPE) && defined(ELEMENT_SIZE)

#if ELEMENT_SIZE == 1
#define COND_DATA_TYPE char
#elif ELEMENT_SIZE == 2
#define COND_DATA_TYPE short
#elif ELEMENT_SIZE == 4
#define COND_DATA_TYPE int
#else // ELEMENT_SIZE
#error "Element size not support"
#endif // ELEMENT_SIZE

#if defined(CONVOLVED_WIDTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(SRC_DEPTH) && defined(PAD_LEFT) && defined(PAD_RIGHT) && defined(PAD_TOP) && defined(PAD_BOTTOM) && defined(PAD_VALUE)
#if defined(DILATION_X) && defined(DILATION_Y)
/** This opencl kernel performs a generic im2col implementation when the data layout is NCHW
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width and height of the input tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT: e.g. -DSRC_WIDTH=128 and -DSRC_HEIGHT=128
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel width, height and depth must be passed at compile time using -DKERNEL_WIDTH, -DKERNEL_HEIGHT and -DSRC_DEPTH: e.g. -DKERNEL_WIDTH=3, -DKERNEL_HEIGHT=3 and -DSRC_DEPTH=64
 * @note The pad_left, pad_right, pad_top and pad_bottom must be passed at compile time using -DPAD_LEFT, -DPAD_RIGHT, -DPAD_TOP and -DPAD_BOTTOM: e.g. -DPAD_LEFT=1, -DPAD_RIGHT=2, -DPAD_TOP=3 and -DPAD_BOTTOM=2
 * @note The zero value to store in case we load values out-of-bounds must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0.0
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note The dilation_x and dilation_y must be passed at compile time using -DDILATION_X and -DDILATION_Y: e.g. -DDILATION_X=1, -DDILATION_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 * @note In case grouping is performed, the number of groups must be passed at compile time using -DNUM_GROUPS: e.g. -DNUM_GROUPS=4
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QASYMM8_SIGNED/QASYMM8/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col_generic_nchw(
    TENSOR3D_DECLARATION(src),
#if defined(NUM_GROUPS)
    TENSOR3D_DECLARATION(dst),
#else  // defined(NUM_GROUPS)
    IMAGE_DECLARATION(dst),
#endif // defined(NUM_GROUPS)
    uint src_stride_w,
    uint dst_stride_w)
{
    const int xc    = get_global_id(0);             // x coordinate in the convolved tensor
    const int yc    = get_global_id(1);             // y coordinate in the convolved tensor
    const int ch    = get_global_id(2) % SRC_DEPTH; // input feature map
    const int batch = get_global_id(2) / SRC_DEPTH; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X - PAD_LEFT;
    const int yi = yc * STRIDE_Y - PAD_TOP;

    // Calculate output indices
#if defined(NUM_GROUPS)
    const int xo = (ch % (SRC_DEPTH / NUM_GROUPS)) * KERNEL_WIDTH * KERNEL_HEIGHT;
    const int zo = ch / (SRC_DEPTH / NUM_GROUPS);
#else                                         // defined(NUM_GROUPS)
    const int xo                   = ch * KERNEL_WIDTH * KERNEL_HEIGHT;
#endif                                        // defined(NUM_GROUPS)
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + ch * src_stride_z + batch * src_stride_w;
#if defined(NUM_GROUPS)
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + yo * dst_stride_y + zo * dst_stride_z + batch * dst_stride_w)) + xo;
#else  // defined(NUM_GROUPS)
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + yo * dst_stride_y + batch * dst_stride_w)) + xo;
#endif // defined(NUM_GROUPS)

    // Linearize convolution elements
    for(int yk = 0; yk < KERNEL_HEIGHT; ++yk)
    {
        int y = yi + yk * DILATION_Y;
        for(int xk = 0; xk < KERNEL_WIDTH; ++xk, ++output_ptr)
        {
            int x = xi + xk * DILATION_X;
#if PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
            *output_ptr = *((__global DATA_TYPE *)(input_ptr + x * src_stride_x + y * src_stride_y));
#else  // PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                *output_ptr = PAD_VALUE;
            }
            else
            {
                *output_ptr = *((__global DATA_TYPE *)(input_ptr + x * src_stride_x + y * src_stride_y));
            }
#endif // PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
        }
    }

#ifdef HAS_BIAS
#if defined(NUM_GROUPS)
    if((xo / (KERNEL_WIDTH * KERNEL_HEIGHT)) == (SRC_DEPTH / NUM_GROUPS - 1))
#else  // defined(NUM_GROUPS)
    if(ch == (SRC_DEPTH - 1))
#endif // defined(NUM_GROUPS)
    {
        *output_ptr = 1.0f;
    }
#endif // HAS_BIAS
}

#endif // defined(DILATION_X) && defined(DILATION_Y)
#endif // defined(CONVOLVED_WIDTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(SRC_DEPTH) && defined(PAD_LEFT) && defined(PAD_RIGHT) && defined(PAD_TOP) && defined(PAD_BOTTOM) && defined(PAD_VALUE) && defined(VECTOR_SIZE) && defined(BOUNDARY_VECTOR_SIZE)
#endif // defined(DATA_TYPE) && defined(ELEMENT_SIZE)
