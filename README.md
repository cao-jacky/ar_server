## ar_server

The server component of the mobile\_ar\_system - single\_process branch is the caching and recognition processes combined into one "server instance" 

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
$ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .
$ make
$ cd ../../build # rerun from this point if changes are made to server.cpp or reco.cpp
$ make
$ cd ..
$ ./gpu_fv size[s/m/l] nearest\_neighbour\_number[1/2/3/4/5] port\_number[#XXXXX] 
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



