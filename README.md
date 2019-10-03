## ar_server

The server component of the mobile\_ar\_system


### Dependencies

  - `OpenCV` , tested with ver 3.3
  - `Cuda` , tested with ver 8.0
  - `falconn` , cpu lsh
  - `CudaSift` , Cuda version of Sift with detection, extraction, matching
  - `Eigen` , Dense data structure
  - `vlfeat` , cpu gmm training

### Installation/Running

Besides the libraries contained in this repo, you may first install `OpenCV` and `Cuda`. 
To build and run:

```sh
$ cd lib/cudasift 
$ sed -i 's/executable/library/g' CMakeLists.txt
$ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .
$ make
$ cd ../../build # rerun from this point if changes are made to server.cpp or reco.cpp
$ make
$ cd ..
$ ./gpu_fv server[s/c] size[s/m/l] nearest\_neighbour\_number[1/2/3/4/5] port\_number[#XXXXX] 
```

### Deployment specifications

The servers were ran on a machine with the following specs:

- CPU: Intel Core i7-8700 3.20GHz x 12
- GPU: GeForce RTX 2080 Ti
- Memory: 32GB
- OS: Ubuntu 18.04.3 LTS

Plus the program was compiled with the following CUDA/GCC/G++ compilers:

- CUDA: 9.2
- GCC: 7.4.0
- G++: 7.4.0

### Testing notes

- 031019: Disabled the cache server since somewhere in the caching pipeline, the cache server would crash. 



