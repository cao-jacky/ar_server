TARGET	= main
CXXSRCS	= $(wildcard ../src/*.cpp)
CSRCS	= $(wildcard ../lib/vlfeat/vl/*.c)
CHEADERS	= $(wildcard ../src/*.h)
CXXHEADERS = $(wildcard ../lib/vlfeat/vl/*.h)
CXXDIR	= $(notdir $(CXXSRCS))
CDIR = $(notdir $(CSRCS))
CXXOBJECTS = $(patsubst %.cpp,%.o, $(CXXDIR)) 
COBJECTS = $(patsubst %.c,%.o, $(CDIR)) 

SIFT_ROOT = ../lib/cudasift
EIGEN_ROOT = ../lib/eigen
FALCONN_ROOT = ../lib/falconn
VL_ROOT = ../lib/vlfeat

INC += -I$(SIFT_ROOT) -I$(EIGEN_ROOT) -I$(FALCONN_ROOT)/src/include -I$(VL_ROOT) -I/usr/include/opencv

LIB = -lm -lrt -lpthread -ldl -pipe -L$(SIFT_ROOT) -lcudasift -std=c++0x -pthread 
LIB += -lcudart -lcufft 
LIB += `pkg-config --cflags --libs opencv`
LIB += -fopenmp -lcublas -lenet
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio

NVCC	= /usr/local/cuda/bin/nvcc -arch=sm_30 -m 64 $(INC) 

CXX = g++ -pipe
CC = gcc -pipe
LINK	= -O3 -pthread 
CFLAGS	= -O3 -c -pthread -fopenmp -DVL_DISABLE_AVX
CXXFLAGS= -O3 -c -pthread -std=c++0x -fopenmp -DVL_DISABLE_AVX

server: $(COBJECTS) $(CXXOBJECTS) cuda_files.o
	$(CXX) $(LINK) -o ../$@ $(COBJECTS) $(CXXOBJECTS) cuda_files.o $(LIB) 

cuda_files.o: ../src/cuda_files.cu 
	$(NVCC) -I$(CUDA_ROOT)/include/  -O3 -c ../src/cuda_files.cu -lcublas 

$(COBJECTS): $(CSRCS) $(CHEADERS) Makefile
	$(CC) $(INC) $(CFLAGS) $(CSRCS) 

$(CXXOBJECTS): $(CXXSRCS) $(CXXHEADERS) Makefile
	$(CXX) $(INC) $(CXXFLAGS) $(CXXSRCS)

clean:
	rm -f ../server $(COBJECTS) $(CXXOBJECTS) cuda_files.o kernels.o

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.

