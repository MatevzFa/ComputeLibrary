
#include "helpers.h"

__kernel void hpvm_interpolate_row(
    TENSOR3D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    uint src_stride_width,
    uint dst_h,
    uint channels,
    uint batches,
    uint perfrow_start,
    uint perfrow_every
    //
)
{
    int x          = get_global_id(0); // Same
    int yo         = get_global_id(1); // Needs mapping
    int chan_batch = get_global_id(2); // Same

    int chan  = chan_batch % channels;
    int batch = chan_batch / channels;

    int src_chan_offset  = chan * src_stride_y;
    int src_batch_offset = batch * src_stride_z;

    int dst_row_offset   = yo * dst_stride_y;
    int dst_chan_offset  = chan * dst_stride_z;
    int dst_batch_offset = batch * dst_stride_w;

    __global uchar *src_chan_ptr    = (src_ptr + src_offset_first_element_in_bytes + src_batch_offset + src_chan_offset);
    __global DATA_TYPE *dst_row_ptr = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + dst_batch_offset + dst_chan_offset + dst_row_offset);

    int needs_interpolation = 0;
    int yi_0 = -1, yi_1 = -1;
    if(yo <= perfrow_start)
    {
        // This row exist in src and no preivous row were perforated out
        yi_0 = yo;
    }
    else if((yo - perfrow_start + 1) % perfrow_every == 0)
    {
        // This row was perforated out
        // Average yi_0 and yi_1

        // This many previous rows are missing
        int offset          = (yo - perfrow_start) / perfrow_every;
        yi_1                = yo - offset;
        yi_0                = yi_1 - 1;
        needs_interpolation = (yo != dst_h - 1);
    }
    else
    {
        // This row exist in src but has to be offset due to some previous rows being perforated out
        int offset = (yo - perfrow_start) / perfrow_every;
        yi_0       = yo - offset;
    }

    __global DATA_TYPE *src0_row_ptr = (__global DATA_TYPE *)(src_chan_ptr + yi_0 * src_stride_width);
    __global DATA_TYPE *src1_row_ptr = (__global DATA_TYPE *)(src_chan_ptr + yi_1 * src_stride_width);

    DATA_TYPE dst_val = src0_row_ptr[x];

    if(needs_interpolation)
    {
        dst_val += src1_row_ptr[x];
        dst_val /= 2;
    }

    dst_row_ptr[x] = dst_val;
}
