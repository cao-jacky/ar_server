## ar_server

The server component of the mobile\_ar\_system - single\_process branch is the caching and recognition processes combined into one "server instance" 

### Docker deployment

Command for running the Docker image is

```sh
$ sudo docker run --entrypoint=/bin/bash --rm --gpus all --net=host -it ghcr.io/giobart/arpipeline -c '/home/ar_server/server lsh 192.168.1.102 false'
$ sudo docker run --entrypoint=/bin/bash --rm --gpus all --net=host -it ghcr.io/giobart/arpipeline
$ sudo docker run --entrypoint=/bin/bash --rm --gpus all --net=host -it ar_server:20220928_1401
```

To compile for multiple architectures:

sudo docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 -t ghcr.io/cao-jacky/ar_server:20220426_1109 --push .

### Dependencies

  - `OpenCV` - tested with version 4.1
  - `CUDA` - tested with version 10.1
  - `FALCONN` - CPU LSH
  - `CudaSift` - CUDA version of SIFT with detection, extraction, matching
  - `Eigen` - Dense data structure
  - `VLFeat` - CPU GMM training

### Installation/Running

Prerequisites are for OpenCV and CUDA to be pre-installed before compiling the code

```sh
$ cd lib/cudasift 
$ sed -i 's/executable/library/g' CMakeLists.txt
$ cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .
$ make
$ cd ../../src/build # rerun from this point if changes are made to server.cpp or reco.cpp
$ make
$ cd ..
$ ./gpu_fv size[s/m/l] nearest_neighbour_number[1/2/3/4/5] port_number[#XXXXX] 
```

### Deployment specifications

The servers were ran on a machine with the following specs:

- CPU: Intel Core i7-8700 3.20GHz x 12
- GPU: GeForce RTX 2080 Ti
- Memory: 32GB
- OS: Ubuntu 18.04.3 LTS

Plus the program was compiled with the following CUDA/GCC/G++ compilers:

- CUDA: 10.1
- GCC: 7.4.0
- G++: 7.4.0

### Testing notes

- 181019: Combined the cache processes and the recognition processes into one "server" - caching now appears to work 

- 031019: Disabled the cache server since somewhere in the caching pipeline, the cache server would crash. 



