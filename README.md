# k-means-clustering
One of my academic projects.

The aim of the project was to implement K-means clustering algorithm using CUDA in two ways - in particular the stage of calculating new coordinates of centroids (centers of clusters).

Directory [k-means-clustering](k-means-clustering) contains three implementations of k-means clustering algorithm:

- [standard sequential CPU algorithm](k-means-clustering/k_means_cpu.hpp) based on http://users.eecs.northwestern.edu/~wkliao/Kmeans/index.html,
- [1st parallel CUDA implementation](k-means-clustering/k_means_gpu_1.cuh) which includes reduction based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf,
- [2nd parallel CUDA implementation](k-means-clustering/k_means_gpu_2.cuh) which is more straightforward than the first CUDA implementation and for reduction uses only `thrust` library functions.

## Input of the algorithm

- random set of `N` points (millions) in `n`-dimensional space (`n` is small, not greater than 15 in my implementation),
- `k` - requested number of clusters, up to several dozen,
- `threshold` - see http://users.eecs.northwestern.edu/~wkliao/Kmeans/index.html.

## Samples

You can find samples in [samples](samples) directory.
The samples use `generate_sample_input` function from [samples utils](samples/common-utils/utils.cpp) to generate pseudo-random input. Generated points are not completely random, they are generated nearby fixed points (they are somewhat clustered).

As for now, there are two samples: [sample-200K-points](samples/sample-200K-points) and [sample-10M-points](sample-10M-points).
For the sample you want you can generate appropiate `Makefile` using `cmake .` command (`CMakeLists.txt` are provided for samples) and then build the executable using `make` command.

If you are not using provided `CMakeLists.txt` (for example you want to compile a sample in MS VS) remember to set proper include directories or to change headers in `#include` in samples' source files to relative paths.

## Example output:

On the test setup (i7 4790k, GTX960 4GB) running provided samples gave the following results:

#### [200K-points](samples/sample-200K-points)

```
N=200000 , n=3 , k=60 , threshold=0.0001

k_means_gpu_1::Compute:
Measured time: 483ms 124us

k_means_gpu_2::Compute:
Measured time: 125ms 375us

k_means_cpu::Compute:
Measured time: 142737ms 19us

Cpu result with Gpu1 comparison: true
Gpu1 result with Gpu2 result comparison: true
```

#### [10M-points](samples/sample-10M-points)

This sample does not run CPU version because it would take unacceptable amount of time.

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