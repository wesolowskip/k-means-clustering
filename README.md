# k-means-clustering
One of my academic projects in CUDA.

The aim of the project was to implement K-means clustering algorithm using CUDA in two ways - in particular the stage of calculating new coordinates of centroids (centers of clusters).

Directory [k-means-clustering](k-means-clustering) contains three implementations of k-means clustering algorithm:

- [standard sequential CPU algorithm](k-means-clustering/k_means_cpu.hpp) based on http://users.eecs.northwestern.edu/~wkliao/Kmeans/index.html,
- [1st parallel CUDA implementation](k-means-clustering/k_means_gpu_1.cuh) which includes reduction based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf,
- [2nd parallel CUDA implementation](k-means-clustering/k_means_gpu_2.cuh) which is more straightforward than the first CUDA implementation and for reduction uses only `thrust` library functions.

## Input of the algorithm

- random set of `N` points (millions) in `n`-dimensional space (`n` should be small, not greater than 15 in my implementation),
- `k` - requested number of clusters, up to several dozen,
- `threshold` - see http://users.eecs.northwestern.edu/~wkliao/Kmeans/index.html.

## Basic example of use

```cpp
//Suppose we have 4 points in 2-dimensional space with coordinates
//  (100, 50), (0, 0), (-20, -7), (-10, 9).
//We want to spit them into 3 clusters.
//We create arrays for cpu and gpu k-means implementations.
//Notice different order of items.
double cpu_data[8] = { 
    100, 50,
      0,  0,
    -20, -7,
    -10,  9
};
double gpu_data[8] = {
    100, 0, -20, -10,
     50, 0,  -7,   9
};

//Now we can run k-means clustering. 
//Of course cpu_data and gpu_data are also iterators. The last argument is threshold.
//Results are of type pair<thrust::host_vector<double>, thrust::host_vector<int>>
auto cpu_res = k_means_cpu::Compute(cpu_data, 4, 2, 3, 0.1);
auto gpu_res_1 = k_means_gpu_1::Compute(gpu_data, 4, 2, 3, 0.1);
auto gpu_res_2 = k_means_gpu_2::Compute(gpu_data, 4, 2, 3, 0.1);

//Now cpu_res.first is:
//  100, 50, -5, 4.5, -20, -7
//and both gpu_res_1.first and gpu_res_2.first are:
//  100, -5, -20, 50, 4.5, -7.
//Note that order of coordinates for these results is same as order of
//coordinates of input points in corresponding functions.
//In particular calculated centroids (centers of clusters) are:
//  (100, 50), (-5, 4.5), (-20, -7).
//All three vectors cpu_res.second, gpu_res_1.second and gpu_res_2.second are:
//  0, 1, 2, 1
//and it means that first input point has been assigned to 0th centroid (100, 50),
//second and fourth input point have been assigned to 1st centroid (-5, 4.5)
//and third point has been assigned to 2nd centroid (-20, -7).
```


## Samples

You can find samples in [samples](samples) directory.
The samples use `generate_sample_input` function from [samples utils](samples/common-utils/utils.cpp) to generate pseudo-random input. Generated points are not completely random, they are generated nearby fixed points (they are somewhat clustered).

As for now, there are two samples: [sample-200K-points](samples/sample-200K-points) and [sample-10M-points](sample-10M-points).
For the sample you want you can generate appropiate `Makefile` using `cmake .` command (`CMakeLists.txt` are provided for samples) and then build the executable using `make` command.

If you are not using provided `CMakeLists.txt` (for example you want to compile a sample in MS VS) remember to set proper include directories or to change headers in `#include` in samples' source files to relative paths.

## Example output:

On the test setup (i7 4790k, GTX960 4GB) running provided samples gave the following results:

#### [200K points](samples/sample-200K-points)

```
N=200000 , n=3 , k=60 , threshold=0.0001

k_means_gpu_1::Compute:
Measured time: 483ms 124us

k_means_gpu_2::Compute:
Measured time: 125ms 375us

k_means_cpu::Compute:
Measured time: 142737ms 19us

Cpu result with Gpu1 result comparison: true
Gpu1 result with Gpu2 result comparison: true
```

#### [10M points](samples/sample-10M-points)

This sample does not run CPU version because it would take an unacceptable amount of time.

```
N=10000000 , n=3 , k=80 , threshold=1e-06

k_means_gpu_1::Compute:
Measured time: 14548ms 97us

k_means_gpu_2::Compute:
Measured time: 8335ms 880us

Gpu1 result with Gpu2 result comparison: true
```

---

Note that provided samples generate pseudo-random data points in each run so the results (timings) of subsequent executions may be a lot different (depending on generated data).
