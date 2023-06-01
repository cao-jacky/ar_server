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

project(HelloWorld C CXX)

set(CMAKE_CXX_STANDARD 11)

include(../cmake/common.cmake)

find_package(OpenCV REQUIRED)

# Proto file
# get_filename_component(hw_proto "protos/helloworld.proto" ABSOLUTE)
# get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
# set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
# set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
# set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
# set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
# add_custom_command(
#       OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
#       COMMAND ${_PROTOBUF_PROTOC}
#       ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${hw_proto_path}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${hw_proto}"
#       DEPENDS "${hw_proto}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fopenmp -DVL_DISABLE_AVX")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -pthread -fopenmp -DVL_DISABLE_AVX")

set(SIFT_ROOT ../lib/cudasift)
set(EIGEN_ROOT ../lib/eigen)
set(FALCONN_ROOT ../lib/falconn)
set(VL_ROOT ../lib/vlfeat)
set(CUDA_HOME /usr/local/cuda-12.1)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${SIFT_ROOT} ${EIGEN_ROOT} ${FALCONN_ROOT}/src/include ${VL_ROOT})

# hw_grpc_proto
# add_library(hw_grpc_proto
#   ${hw_grpc_srcs}
#   ${hw_grpc_hdrs}
#   ${hw_proto_srcs}
#   ${hw_proto_hdrs})
# target_link_libraries(hw_grpc_proto
#   ${_REFLECTION}
#   ${_GRPC_GRPCPP}
#   ${_PROTOBUF_LIBPROTOBUF})

link_directories(${SIFT_ROOT} ${CUDA_HOME}/lib)

add_executable(server ../src/cuda_files.cu ${CXXSRCS} ${CSRCS})
set_target_properties(server PROPERTIES LINKER_LANGUAGE CXX)

# target_link_libraries(server cublas cudasift cudart cufft ${OpenCV_LIBS} m rt pthread dl)
target_link_libraries(server cublas cudasift cudart cufft ${OpenCV_LIBS} opencv_core opencv_highgui opencv_imgproc opencv_videoio m rt pthread dl)

# Targets greeter_[async_](client|server)
# foreach(_target
#   greeter_client greeter_server)
#   add_executable(${_target} "${_target}.cc")
#   target_link_libraries(${_target}
#     hw_grpc_proto
#     absl::flags
#     absl::flags_parse
#     ${_REFLECTION}
#     ${_GRPC_GRPCPP}
#     ${_PROTOBUF_LIBPROTOBUF})
# endforeach()
