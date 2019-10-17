TARGET	= main
CXXSRCS	= $(wildcard ../src/*.cpp)
CSRCS	= $(wildcard ../lib/vlfeat/vl/*.c)
CHEADERS	= $(wildcard ../src/*.h)
CXXHEADERS = $(wildcard ../lib/vlfeat/vl/*.h)
CXXDIR	= $(notdir $(CXXSRCS))
CDIR = $(notdir $(CSRCS))
CXXOBJECTS = $(patsubst %.cpp,%.o, $(CXXDIR)) 
COBJECTS = $(patsubst %.c,%.o, $(CDIR)) 

CUDA_ROOT = /usr/local/cuda

SIFT_ROOT = ../lib/cudasift
EIGEN_ROOT = ../lib/eigen
FALCONN_ROOT = ../lib/falconn
VL_ROOT = ../lib/vlfeat

INC += -I$(SIFT_ROOT) -I$(EIGEN_ROOT) -I$(FALCONN_ROOT)/src/include -I$(VL_ROOT)

LIB = -lm -lrt -lpthread -ldl -pipe -L$(SIFT_ROOT) -lcudasift -std=c++0x -pthread 
LIB += -L$(CUDA_ROOT)/lib64/ -lcudart -lcufft 
LIB += `pkg-config --cflags --libs opencv`
LIB += -fopenmp -lcublas

#NVCC	= /usr/bin/nvcc -arch=sm_75 -m 64 $(INC) 
NVCC    = /usr/local/cuda-10.1/bin/nvcc -arch=sm_75 -G -g -m 64 $(INC)
#NVCC	= $(CUDA_ROOT)/bin/nvcc -arch=sm_20 -m 64 $(INC)

CXX = g++ -pipe
CC = gcc -pipe
LINK	= -O3 -pthread 
CFLAGS	= -O3 -c -pthread -fopenmp -DVL_DISABLE_AVX
CXXFLAGS= -O3 -c -pthread -std=c++0x -fopenmp -DVL_DISABLE_AVX

gpu_fv: $(COBJECTS) $(CXXOBJECTS) cuda_files.o
	$(CXX) $(LINK) -o ../$@ $(COBJECTS) $(CXXOBJECTS) cuda_files.o $(LIB) 

cuda_files.o: ../src/cuda_files.cu 
	$(NVCC) -I$(CUDA_ROOT)/include/  -O3 -c ../src/cuda_files.cu -lcublas 

$(COBJECTS): $(CSRCS) $(CHEADERS) Makefile
	$(CC) $(INC) $(CFLAGS) $(CSRCS) 

$(CXXOBJECTS): $(CXXSRCS) $(CXXHEADERS) Makefile
	$(CXX) $(INC) $(CXXFLAGS) $(CXXSRCS)

clean:
	rm -f ../gpu_fv $(COBJECTS) $(CXXOBJECTS) cuda_files.o kernels.o

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.

