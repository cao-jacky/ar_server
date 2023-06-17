FROM ghcr.io/giobart/active-internal-queue/active-sidecar-queue:v1.0.5-cuda12


# Install necessary dependencies
RUN apt-get update && apt-get -y install cmake protobuf-compiler libprotobuf-dev

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Clone OpenCV repository
RUN git clone https://github.com/opencv/opencv.git

# Build and install OpenCV
WORKDIR /opencv/build
RUN cmake ..
RUN make -j$(nproc)
RUN make install

# Install Protobuf manually
RUN apt-get install wget
WORKDIR /
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protobuf-cpp-3.17.3.tar.gz \
    && tar -zxvf protobuf-cpp-3.17.3.tar.gz \
    && rm protobuf-cpp-3.17.3.tar.gz
WORKDIR /protobuf-3.17.3
RUN ./configure \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Clone gRPC repository
WORKDIR /
RUN git clone --recurse-submodules -b v1.56.0 https://github.com/grpc/grpc.git

# Build and install gRPC
WORKDIR /grpc
RUN mkdir -p cmake/build
WORKDIR /grpc/cmake/build
RUN cmake ../.. && make -j$(nproc) && make install

# Set library search path
RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf
RUN ldconfig

# Set environment variables
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/local/lib


ADD lib /home/lib

RUN cd /home/lib/cudasift &&  sed -i 's/executable/library/g' CMakeLists.txt && cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release . && make

#Install Json lib
RUN apt-get update && apt-get -y install nlohmann-json3-dev

# Add and build final sources
ADD src /home/src
ADD cmake /home/cmake
ADD data /data

RUN cd /home/src && mkdir build && cd build && cmake .. && make

RUN mv /data /home/data

WORKDIR /home
