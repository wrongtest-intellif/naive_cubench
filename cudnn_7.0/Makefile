CUDA_HOME = /usr/local/cuda-9.0
CUDNN_HOME = /home/wrongtest/tools/cudnn-7.0
GFLAGS_HOME = /home/wrongtest/tools/gflags/build

CUDA_FLAGS = -I$(CUDNN_HOME)/include -lcudnn -L$(CUDNN_HOME)/lib64 \
             -I$(CUDA_HOME)/include -lcudart -L$(CUDA_HOME)/lib64 
GFLAGS_FLAGS = -I$(GFLAGS_HOME)/include -L$(GFLAGS_HOME)/lib -lgflags
DEP_FLAGS = -I.. $(CUDA_FLAGS) $(GFLAGS_FLAGS) 


all: cnn_forward cnn_forward_fused

cnn_forward:
	nvcc -std=c++11 $(DEP_FLAGS) cnn_forward.cpp -o cnn_forward -O2

cnn_forward_fused:
	nvcc -std=c++11 $(DEP_FLAGS) cnn_forward_fused.cpp -o cnn_forward_fused -O2

clean:
	rm -f cnn_forward cnn_forward_fused
