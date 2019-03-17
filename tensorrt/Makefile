TENSORRT_HOME = /home/wrongtest/tools/TensorRT-5.0.0.10
GFLAGS_HOME = /home/wrongtest/tools/gflags/build

GFLAGS_FLAGS = -I$(GFLAGS_HOME)/include -L$(GFLAGS_HOME)/lib -lgflags
DEP_FLAGS = -I.. -I$(TENSORRT_HOME)/include -L$(TENSORRT_HOME)/lib $(GFLAGS_FLAGS)


all: cnn_forward cnn_forward_fused

cnn_forward: cnn_forward.cpp
	nvcc -std=c++11 $(DEP_FLAGS) -lnvinfer cnn_forward.cpp -o cnn_forward -O2

cnn_forward_fused: cnn_forward_fused.cpp
	nvcc -std=c++11 $(DEP_FLAGS) -lnvinfer cnn_forward_fused.cpp -o cnn_forward_fused -O2

clean:
	rm -f cnn_forward cnn_forward_fused

