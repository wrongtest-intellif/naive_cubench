#ifndef __TENSORRT_UTILS_H__
#define __TENSORRT_UTILS_H__

#include <NvInfer.h>
#include <NvUtils.h>


class TensorRTLogger : public nvinfer1::ILogger {
public:
    TensorRTLogger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override {
        if (severity > reportableSeverity) return;
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        	printf("INTERNAL_ERROR: %s\n", msg); break;
        case Severity::kERROR: 
        	printf("ERROR: %s\n", msg); break;
        case Severity::kWARNING:
        	printf("WARNING: %s\n", msg); break;
        case Severity::kINFO:
        	printf("INFO: %s\n", msg); break;
        default:
        	printf("UNKNOWN: %s\n", msg); break;
        }
    }
    Severity reportableSeverity;
};


#endif  // __TENSORRT_UTILS_H__
