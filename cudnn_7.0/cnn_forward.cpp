#include <string>
#include <limits>
#include <chrono>

#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "gflags/gflags.h"
#include "cuda_utils.h"


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
DEFINE_int32(iters, 10000, "compute iterations");


// algorithms
const std::string algos[] = {
    "implicit",             // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM=0
    "implicit_precompute",  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM=1
    "explicit",             // CUDNN_CONVOLUTION_FWD_ALGO_GEMM=2
    "direct",               // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT=3
    "fft",                  // CUDNN_CONVOLUTION_FWD_ALGO_FFT=4
    "fft_tiling",           // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING=5
    "winograd",             // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD=6
    "winograd_nonfused",    // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED=7
};


int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
 
    float alpha = 1.0f, beta = 0.0f;

    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t result_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    if (FLAGS_dtype == "float") {
        data_type = CUDNN_DATA_FLOAT;
        result_type = CUDNN_DATA_FLOAT;
        data_format = CUDNN_TENSOR_NCHW;
    } else if (FLAGS_dtype == "int8") {
        data_type = CUDNN_DATA_INT8x4;
        result_type = CUDNN_DATA_INT32;
        data_format = CUDNN_TENSOR_NCHW_VECT_C;
    }

    cudnnTensorDescriptor_t x_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        x_desc, data_format, data_type, 
        FLAGS_batch_size, FLAGS_input_channels, FLAGS_height, FLAGS_width));
 
    cudnnFilterDescriptor_t w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        w_desc, data_type, data_format, 
        FLAGS_output_channels, FLAGS_input_channels, 
        FLAGS_kernel_height, FLAGS_kernel_width));

    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolutionMathType(conv_desc, 
    	CUDNN_DEFAULT_MATH));
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, 1));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
    	FLAGS_padding_height, FLAGS_padding_width, 
        FLAGS_stride_height, FLAGS_stride_width, 
        FLAGS_dilation_height, FLAGS_dilation_width,
        CUDNN_CONVOLUTION, result_type));

    int n, c, height, width;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc,
    	x_desc, w_desc, &n, &c, &height, &width));
    printf("Intput dimensions: [%d, %d, %d, %d]\n", 
        FLAGS_batch_size, FLAGS_input_channels, FLAGS_height, FLAGS_width);
    printf("Kernel dimensions: [%d, %d, %d, %d]\n",
        FLAGS_output_channels, FLAGS_input_channels, FLAGS_kernel_height, FLAGS_kernel_width);
    printf("Output dimensions: [%d, %d, %d, %d]\n", n, c, height, width);

    cudnnTensorDescriptor_t y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        y_desc, data_format, data_type, n, c, height, width));

    float min_cost = std::numeric_limits<float>::max();
    cudnnConvolutionFwdAlgo_t algo = cudnnConvolutionFwdAlgo_t(0);
    if (FLAGS_algo == "") {
        printf("No algorithm specified, try algorithms tuning\n");
        int algo_found;
        cudnnConvolutionFwdAlgoPerf_t perf[32];
        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
            x_desc, w_desc, conv_desc, y_desc, 32, &algo_found, perf));
        for (int i = 0; i < algo_found; ++i) {
            if (perf[i].time > 0) {
                printf("%s: time=%f mem=%lu det=%d dtype=%d\n",
                    algos[perf[i].algo].c_str(), perf[i].time, perf[i].memory,
                    perf[i].determinism, perf[i].mathType);
                if (perf[i].time < min_cost) {
                    min_cost = perf[i].time;
                    algo = perf[i].algo;
                }
            }
            if (min_cost == std::numeric_limits<float>::max()) {
                fprintf(stderr, "No available algorithm\n");
                exit(1);
            }
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            if (algos[i] == FLAGS_algo) {
                algo = cudnnConvolutionFwdAlgo_t(i);
                break;
            }
            if (i == 8 - 1) {
                fprintf(stderr, "Unknown algorithm: %s\n", FLAGS_algo.c_str());
                exit(1);
            }
        }
    }
    printf("Use algorithm %s: %d\n", algos[algo].c_str(), algo);
   
    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
        x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));
    printf("Workspace byte size: %lu\n", workspace_size);

    float *x, *y, *w, *workspace;
    CHECK_CUDA_RUNTIME(cudaMalloc(&x, sizeof(float) *
        FLAGS_batch_size * FLAGS_input_channels * FLAGS_height * FLAGS_width));
    CHECK_CUDA_RUNTIME(cudaMalloc(&y, sizeof(float) * 
        n * c * height * width));
    CHECK_CUDA_RUNTIME(cudaMalloc(&w, sizeof(float) * 
        FLAGS_output_channels * FLAGS_input_channels * 
        FLAGS_kernel_height * FLAGS_kernel_width));
    CHECK_CUDA_RUNTIME(cudaMalloc(&workspace, workspace_size));
    
    // warmup
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        y_desc,
        y));

    auto begin = std::chrono::system_clock::now();
    for (int i = 0; i < FLAGS_iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn_handle,
            &alpha,
            x_desc,
            x,
            w_desc,
            w,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            &beta,
            y_desc,
            y));
    }
    auto end = std::chrono::system_clock::now();
    int64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - begin).count();
    printf("Milliseconds per iter: %f\n", time / (double)FLAGS_iters);
    return 0;
}
