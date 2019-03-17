#include <stdio.h>
#include <stdlib.h>
#include <NvInfer.h>
#include <NvUtils.h>
#include <cuda_runtime.h>

#include <vector>
#include <chrono>
#include <algorithm>

#include "utils/tensorrt_utils.h"
#include "utils/cnn_forward_flags.h"

using namespace nvinfer1;


static TensorRTLogger gLogger;


int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    DataType dataType = DataType::kFLOAT;
    IBuilder* builder = createInferBuilder(gLogger);
    if (FLAGS_dtype == "int8") {
        builder->setInt8Mode(true);
        builder->setStrictTypeConstraints(true);
        // builder->setInt8Calibrator(/* */);
    }
    INetworkDefinition* network = builder->createNetwork();

    auto data = network->addInput(
        "data", dataType,
        Dims4{FLAGS_batch_size, FLAGS_input_channels,
              FLAGS_height, FLAGS_width});

    Weights filter, bias;
    filter.type = dataType;
    filter.count = FLAGS_kernel_height * FLAGS_kernel_width * \
                   FLAGS_input_channels * FLAGS_output_channels;
    filter.values = malloc(sizeof(float) * filter.count);

    bias.type = dataType;
    bias.count = 0;  // FLAGS_output_channels;
    bias.values = nullptr;  // malloc(sizeof(float) * bias.count);

    auto conv = network->addConvolution(
        *data, FLAGS_output_channels,
        DimsHW{FLAGS_kernel_height, FLAGS_kernel_width},
        filter, bias);
    conv->setStride(DimsHW{FLAGS_stride_height, FLAGS_stride_width});
    conv->setDilation(DimsHW{FLAGS_dilation_height, FLAGS_dilation_width});
    conv->setPadding(DimsHW{FLAGS_padding_height, FLAGS_padding_width});
    if (FLAGS_dtype == "int8") {
        conv->setPrecision(DataType::kINT8);
        conv->setOutputType(0, DataType::kFLOAT);
        conv->getOutput(0)->setDynamicRange(0, 128);
        data->setDynamicRange(0, 128);
    }

    network->markOutput(*conv->getOutput(0));

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    IExecutionContext *context = engine->createExecutionContext();

    std::vector<void*> buffers;
    for (int i = 0; i < engine->getNbBindings(); i++) {
        Dims dims = engine->getBindingDimensions(i);
        int64_t count = std::accumulate(
            dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        void* gpu_buffer;
        cudaMalloc(&gpu_buffer, count * sizeof(float));
        buffers.push_back(gpu_buffer);
    }

    int inputIdx = engine->getBindingIndex(data->getName());
    int outputIdx = engine->getBindingIndex(conv->getOutput(0)->getName());

    const int warmup_steps = 10;
    for (int i = 0; i < warmup_steps; ++i) {
        context->execute(FLAGS_batch_size, buffers.data());
    }

    cudaDeviceSynchronize();
    auto begin = std::chrono::system_clock::now();
    for (int i = 0; i < FLAGS_iters; ++i) {
        context->execute(FLAGS_batch_size, buffers.data());
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    int64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - begin).count();
    printf("Milliseconds per iter: %.3f\n", time / (double)FLAGS_iters);

    return 0;
}
