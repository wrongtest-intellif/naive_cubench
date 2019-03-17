#ifndef __CNN_FORWARD_FLAGS_H__
#define __CNN_FORWARD_FLAGS_H__

#include "gflags/gflags.h"


// flags
DEFINE_int32(height, 224, "height of image");
DEFINE_int32(width, 224, "width of image");
DEFINE_int32(input_channels, 256, "input channels");
DEFINE_int32(output_channels, 256, "output channels");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(kernel_height, 3, "height of kernel");
DEFINE_int32(kernel_width, 3, "width of kernel");
DEFINE_int32(padding_height, 1, "padding of height");
DEFINE_int32(padding_width, 1, "padding of width");
DEFINE_int32(stride_height, 1, "stride of height");
DEFINE_int32(stride_width, 1, "stride of width");
DEFINE_int32(dilation_height, 1, "dilation of height");
DEFINE_int32(dilation_width, 1, "dilation of width");
DEFINE_string(dtype, "float", "data type in [float, int8]");
DEFINE_string(algo, "", "cnn algorithm");
DEFINE_int32(iters, 1000, "compute iterations");

// for fused op
DEFINE_int32(enable_fuse, 1, "Whether do a fused computation on needed");
DEFINE_int32(do_bias, 1, "Whether add bias in fused op");
DEFINE_int32(do_relu, 1, "Whether do relu activation in fused op");
DEFINE_int32(do_add, 1, "Whether add bypass part in fused op");

#endif  // __CNN_FORWARD_FLAGS_H__
