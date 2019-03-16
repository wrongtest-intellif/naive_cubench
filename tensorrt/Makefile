TENSORRT_HOME = /home/wrongtest/tools/TensorRT-5.0.0.10
GFLAGS_HOME = /home/wrongtest/tools/gflags/build

GFLAGS_FLAGS = -I$(GFLAGS_HOME)/include -L$(GFLAGS_HOME)/lib -lgflags
DEP_FLAGS = -I.. -I$(TENSORRT_HOME)/include $(GFLAGS_FLAGS)

cnn_forward: cnn_forward.cpp
	nvcc -std=c++11 $(DEP_FLAGS) -L$(TENSORRT_HOME)/lib -lnvinfer cnn_forward.cpp -o cnn_forward -O2


clean:
	rm -f cnn_forward

