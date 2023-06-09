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
file(GLOB TARGET_H CHEADERS "*.h")
file(GLOB TARGET_H CXXHEADERS "../lib/vlfeat/vl/*.h")

set(HEADERS reco.hpp)

cuda_add_library(cuda_files SHARED cuda_files.cu)
target_link_libraries(cuda_files cublas)

add_compile_options(-O3 -pthread -fopenmp -DVL_DISABLE_AVX)

# Proto file
get_filename_component(oakestra_proto "protos/frame.proto" ABSOLUTE)
get_filename_component(oakestra_proto_path "${oakestra_proto}" PATH)

# Generated sources
set(oakestra_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/frame.pb.cc")
set(oakestra_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/frame.pb.h")
set(oakestra_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/frame.grpc.pb.cc")
set(oakestra_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/frame.grpc.pb.h")
add_custom_command(
      OUTPUT "${oakestra_proto_srcs}" "${oakestra_proto_hdrs}" "${oakestra_grpc_srcs}" "${oakestra_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${oakestra_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${oakestra_proto}"
      DEPENDS "${oakestra_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(oakestra_grpc_proto
  ${oakestra_grpc_srcs}
  ${oakestra_grpc_hdrs}
  ${oakestra_proto_srcs}
  ${oakestra_proto_hdrs})
target_link_libraries(oakestra_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

add_library(vlfeat ${VLFEAT_SOURCE_FILES})

add_library(reco STATIC reco.cc)

add_library(primary STATIC services/primary.cc)
add_library(sift STATIC services/sift.cc)
add_library(encoding STATIC services/encoding.cc)
add_library(lsh STATIC services/lsh.cc)
add_library(matching STATIC services/matching.cc)

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

cuda_add_executable(server 
    server.cc
    reco.cc
    ${HEADERS}
    services/primary.cc
    services/sift.cc
    services/encoding.cc
    services/lsh.cc
    services/matching.cc
    cuda_files.cu 
    ${CXXSRCS} 
    ${CSRCS}
    OPTIONS -arch=sm_60
)

set_property(TARGET server 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(reco -L$(SIFT_ROOT) -lcudasift)

target_link_libraries(server
  reco
  primary
  sift
  encoding
  lsh
  matching
  ${OpenCV_LIBS} 
  oakestra_grpc_proto
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