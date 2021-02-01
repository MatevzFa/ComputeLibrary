#include "helpers.h"

#define INDEX(ptr, i) ((__global DATA_TYPE *)ptr)[i]

__kernel void hpvm_add_offset(
    IMAGE_DECLARATION(input), // input matrix
    IMAGE_DECLARATION(accum), // accumulator matrix
    ulong w,                  // width of single matrix
    ulong h,                  // height of single matrix
    ulong hw,                 // length of each row (== h*w)
    ulong m,                  // n of filters
    ulong offset_w,           // horizontal offset (w)
    ulong offset_h            // vertical offset (h)
)
{
    Image _input = CONVERT_TO_IMAGE_STRUCT(input);
    Image _accum = CONVERT_TO_IMAGE_STRUCT(accum);

    ulong idx_hw = get_global_id(0);
    ulong idx_m  = get_global_id(1);

    ulong idx_h = idx_hw / w;
    ulong idx_w = idx_hw % w;

    // Always valid
    ulong idx_dst = (idx_m * hw) + idx_h * w + idx_w;

    // Not always valid
    ulong idx_h_alt = idx_h + offset_h;
    ulong idx_w_alt = idx_w + offset_w;
    ulong idx_src   = (idx_m * hw) + idx_h_alt * w + idx_w_alt;

    if(idx_h_alt < h && idx_w_alt < w)
    {
        INDEX(accum_ptr, idx_dst) += INDEX(input_ptr, idx_src);
    }
    // INDEX(accum_ptr, idx_dst) += (DATA_TYPE)1;
}
