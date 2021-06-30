#ifndef KMEANSCLUSTERING_COMMON_KERNELS_CUH
#define KMEANSCLUSTERING_COMMON_KERNELS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_ptr.h>
#include "constants.h"
#include <limits> //DBL_MAX

namespace common_kernels
{

	/// Initializes k centroids as first k points.
	/// \param points
	/// \param centroids
	/// \param N
	/// \param n
	/// \param k
	template <int n>
    __global__ void initialize_centroids(
            const thrust::device_ptr<double> points,
            thrust::device_ptr<double> centroids,
            const int N,
            const int k
    )
    {
        //Grid-stride-loop
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < k; i += stride)
        {
            //Initializing i-th centroid coordinates
            for (int centroid_coordinate_index = i, point_coordinate_index = i;
                 centroid_coordinate_index < k*n;
                 centroid_coordinate_index += k, point_coordinate_index += N)
                centroids[centroid_coordinate_index] = points[point_coordinate_index];
        }
    }

	/// Function calculates distances between points and centroids and for each point finds index of closest centroid.
	/// \param points
	/// \param centroids
	/// \param closest_centroid_indices
	/// \param N
	/// \param n
	/// \param k
	template <int n>
    __global__ void find_closest_centroids(
            const thrust::device_ptr<double> points,
            const thrust::device_ptr<double> centroids,
            thrust::device_ptr<int> closest_centroid_indices,
            const int N,
            const int k
    )
    {
        //Grid-stride-loop
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < N; i += stride)
        {
            //Calculating distance between i-th point and every centroid.
            //Firstly we store coordinates of i-th point to local array
            int point_coordinates[MAX_n];
            for (int j = 0, point_coordinate_index = i; j < n; j++, point_coordinate_index += N)
                point_coordinates[j] = points[point_coordinate_index];
            double min_squared_distance = DBL_MAX;
            int closest_centroid_index = -1;
            for (int j = 0; j < k; j++)
            {
                //j-th centroid
                double squared_distance = 0.;
                for (int l = 0, centroid_coordinate_index = j;
                     l < n;
                     l++, centroid_coordinate_index += k)
                {
                    double tmp1 = (double)point_coordinates[l];
                    double tmp2 = centroids[centroid_coordinate_index];
                    squared_distance += (tmp1 - tmp2)*(tmp1 - tmp2);
                }
                if (squared_distance < min_squared_distance)
                {
                    min_squared_distance = squared_distance;
                    closest_centroid_index = j;
                }
            }
            closest_centroid_indices[i] = closest_centroid_index;
        }
    }
}

#endif