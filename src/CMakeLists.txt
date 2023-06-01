# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake build file for C++ helloworld example.
# Assumes protobuf and gRPC have been installed using cmake.
# See cmake_externalproject/CMakeLists.txt for all-in-one cmake build
# that automatically builds all the dependencies before building helloworld.

cmake_minimum_required(VERSION 3.8)

project(server LANGUAGES CUDA CXX C)

# file(GLOB_RECURSE CXXSRCS ../*.cpp ../*/*.cpp)

set(SIFT_ROOT ../lib/cudasift)
set(EIGEN_ROOT ../lib/eigen)
set(FALCONN_ROOT ../lib/falconn)
set(VL_ROOT ../lib/vlfeat)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# find_library(cudasift cudasift ../lib/cudasift)
# find_package(VLFeat REQUIRED)

include(../cmake/common.cmake)

include_directories(
    ${SIFT_ROOT}
    ${EIGEN_ROOT}
    ${FALCONN_ROOT}/src/include
    ${VL_ROOT}
    /usr/local/include/opencv4
)

link_directories(
    ${SIFT_ROOT}
    ${EIGEN_ROOT}
    ${FALCONN_ROOT}/src/include
    ${VL_ROOT}
    /usr/local/cuda-12.1/lib64
)

get_filename_component(CURRENT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
message("Current directory: ${CURRENT_DIR}")

file(GLOB TARGET_CPP CXXSRCS "*.cpp" "*/*.cpp" "*.cc" "*/*.cc")

file(GLOB VLFEAT_SOURCE_FILES ../lib/vlfeat/vl/*.c)
# file(GLOB CUDASIFT_SOURCE_FILES ../lib/cudasift/*)

file(GLOB TARGET_H CHEADERS "*.h")
file(GLOB TARGET_H CXXHEADERS "../lib/vlfeat/vl/*.h")


# list(APPEND CUDA_NVCC_FLAGS "-arch=sm_60 -m 64")

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60, code=sm_60)
# cuda_add_library(cuda_files STATIC cuda_files.cu)
# cuda_compile(cuda_files cuda_files.cu)

# cuda_add_library(cudasift STATIC ${SIFT_ROOT}/cudaImage.cu)
cuda_add_library(cuda_files SHARED cuda_files.cu)
target_link_libraries(cuda_files cublas)


add_compile_options(-O3 -pthread -fopenmp -DVL_DISABLE_AVX)
# add_compile_options(-O3 -pthread -fopenmp -DVL_DISABLE_AVX)

# Proto file
get_filename_component(hw_proto "protos/helloworld.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# hw_grpc_proto
add_library(hw_grpc_proto
  ${hw_grpc_srcs}
  ${hw_grpc_hdrs}
  ${hw_proto_srcs}
  ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Targets greeter_[async_](client|server)
# foreach(_target
#   server greeter_client greeter_server)
#   add_executable(${_target} "${_target}.cc")
#   target_link_libraries(${_target}
#     hw_grpc_proto
#     absl::flags
#     absl::flags_parse
#     ${_REFLECTION}
#     ${_GRPC_GRPCPP}
#     ${_PROTOBUF_LIBPROTOBUF})
# endforeach()

add_library(vlfeat ${VLFEAT_SOURCE_FILES})
# add_library(cudasift ${CUDASIFT_SOURCE_FILES})
# add_library(cudasift STATIC ${SIFT_ROOT}/mainSift.cpp)

add_library(reco STATIC reco.cc)

# cuda_add_executable(
#   reco
#   reco.cc
#   cuda_files.cu
# )

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
) 

set(sources
  geomFuncs.cpp  
  mainSift.cpp  
)

# cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_60)

cuda_add_executable(server 
    server.cc
    reco.cc
    cuda_files.cu 
    ${CXXSRCS} 
    ${CSRCS}
    OPTIONS -arch=sm_60
)

set_property(TARGET server 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

             
# target_link_libraries(cudasift -L{SIFT_ROOT})
# target_link_libraries(cudasift -L${SIFT_ROOT} -lcudasift)

target_link_libraries(reco -L$(SIFT_ROOT) -lcudasift)

target_link_libraries(server
  reco
  ${OpenCV_LIBS} 
  hw_grpc_proto
  absl::flags
  absl::flags_parse
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  cudart 
  cufft 
  -lcublas
  opencv_core 
  opencv_highgui 
  opencv_imgproc 
  opencv_videoio 
  cublas 
  enet
  vlfeat
  OpenMP::OpenMP_CXX
  )