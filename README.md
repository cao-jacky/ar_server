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

