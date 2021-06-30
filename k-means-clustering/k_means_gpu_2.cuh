#ifndef KMEANSCLUSTERING_K_MEANS_GPU_2_CUH
#define KMEANSCLUSTERING_K_MEANS_GPU_2_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/discard_iterator.h>
#include "common_kernels.cuh"
#include "constants.h"

using namespace std;
using namespace common_kernels;

namespace k_means_gpu_2
{
    template <int n>
    __global__ void average_centroids(const thrust::device_ptr<int> keys,
                                      const thrust::device_ptr<double> sums,
                                      const thrust::device_ptr<int> counts,
                                      thrust::device_ptr<double> centroids,
                                      int keys_count,
                                      const int k)
    {
        //Grid-stride-loop
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < keys_count; i += stride)
        {
            int which_centroid = keys[i];
            int count = counts[i];
            if (count)
            {
                for (int j = 0, sum_index = i, centroid_index = which_centroid;
                j < n;
                j++, sum_index += keys_count, centroid_index += k)
                    centroids[centroid_index] = sums[sum_index] / (double)count;
            }
        }
    }

    /// Templated implementation of second k-means clustering algorithm using GPU.
    /// For parameter details see Compute function description.
    /// \tparam n
    /// \tparam InputIterator
    /// \param h_input_data
    /// \param N
    /// \param k
    /// \param threshold
    /// \return
    template <int n, class InputIterator>
    pair<thrust::host_vector<double>, thrust::host_vector<int>> templated_compute(
            const InputIterator h_input_data,
            const int N,
            const int k,
            const double threshold
    )
    {
        if (k > N)
            throw invalid_argument("k must be less or equal to N");

        const int blockSize = 1024;
        const int numBlocks_k_threads = (k + blockSize - 1) / blockSize;
        const int numBlocks_N_threads = (N + blockSize - 1) / blockSize;

        thrust::device_vector<double> input_data(h_input_data, h_input_data + N * n);
        thrust::device_vector<double> centroids(k*n);
        thrust::device_vector<int> closest_centroid_indices(N, -1);
        thrust::device_vector<int> prev_closest_centroid_indices(N);
        thrust::device_vector<int> sort_keys(N);
        thrust::device_vector<int> reduced_keys(k);
        thrust::device_vector<int> reduced_counts(k);
        thrust::device_vector<double> reduced_sums(k*n);
        thrust::device_vector<int> sorted_point_indices(N);

        initialize_centroids<n> <<<numBlocks_k_threads, blockSize >>> (input_data.data(), centroids.data(), N, k);

        int delta;
        int iter = 0;
        do
        {
            closest_centroid_indices.swap(prev_closest_centroid_indices);
            find_closest_centroids<n> <<<numBlocks_N_threads, blockSize >>> (
                    input_data.data(),
                    centroids.data(),
                    closest_centroid_indices.data(),
                    N, k);
            //Calculating number of points which assignment to cluster has changed
            delta = thrust::inner_product(
                    prev_closest_centroid_indices.begin(),
                    prev_closest_centroid_indices.end(),
                    closest_centroid_indices.begin(),
                    0,
                    thrust::plus<int>(),
                    thrust::not_equal_to<int>());
            //Sorting indices by keys which are indices of closest centroids
            thrust::sequence(sorted_point_indices.begin(), sorted_point_indices.end());
            thrust::copy(closest_centroid_indices.begin(), closest_centroid_indices.end(), sort_keys.begin());
            thrust::sort_by_key(
                    sort_keys.begin(),
                    sort_keys.end(),
                    sorted_point_indices.begin()
            );
            //Note that some centroids may have no points assigned. In such a situation there is no centroid's
            //index in sort_keys
            int reduced_keys_count = 0;
            int current_offset = 0;
            for (int i = 0; i < n; i++)
            {
                //Reducing subsequent coordinates of points belonging to specific centroids.
                //Firstly, I create iterator that transforms points' indices to corresponding coordinates
                auto sorted_coordinates_it = thrust::make_permutation_iterator(
                        input_data.begin() + i*N,
                        sorted_point_indices.begin()
                );

                if (i == 0)
                {
                    //First iteration - I reduce counts of points belonging to each centroid
                    auto reduced_keys_end = thrust::reduce_by_key(
                            sort_keys.begin(),
                            sort_keys.end(),
                            thrust::make_constant_iterator(1),
                            reduced_keys.begin(),
                            reduced_counts.begin()
                    ).first;
                    //reduced_keys_count describes how many keys are left after reduction (how many clusters have
                    //positive number of assigned points)
                    reduced_keys_count = reduced_keys_end - reduced_keys.begin();
                }
                //Reducing sums of specific coordinate
                thrust::reduce_by_key(
                        sort_keys.begin(),
                        sort_keys.end(),
                        sorted_coordinates_it,
                        thrust::make_discard_iterator(),
                        reduced_sums.begin() + current_offset
                );
                current_offset += reduced_keys_count;
            }
            //Calculating new coordinates of centroids
            const int numBlocks = (reduced_keys_count + blockSize - 1) / blockSize;
            //reduced_keys are indices of consecutive centroids. It is necessary because some centroids may have no
            //points assigned
            average_centroids<n> <<<numBlocks, blockSize >>> (reduced_keys.data(), reduced_sums.data(), reduced_counts.data(),
                                                              centroids.data(), reduced_keys_count, k);
        } while ((double)delta / N > threshold && (iter++ < MAX_ITERS));
        if (iter >= MAX_ITERS)
            printf("Max iteration count reached\n");
        //Copying results to host
        thrust::host_vector<double> h_centroids{ centroids };
        thrust::host_vector<int> h_centroid_assignments{ closest_centroid_indices };
        return std::make_pair(h_centroids, h_centroid_assignments);
    }

    /// Wrapper for second implementation of k-means clustering using GPU.
    /// Using the fact that n has its maximum value we can use template with parameter n (you can find similar
    /// example here https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).
    /// \tparam InputIterator h_input_data type
    /// \param h_input_data Iterator to double coordinates of data points. Corresponding collection should be
    /// of size N*n.
    /// IMPORTANT NOTE! Order of items is different from the order of corresponding parameter in k_means_cpu::Compute function.
    /// Here, first N elements are coordinates of first coordinate of all data points. Next N elements correspond to
    /// second coordinate of all data points and so forth
    /// \param N Number of data points
    /// \param k Requested number of clusters
    /// \param n Dimension of data points, must be less or equal to MAX_n defined in constants header
    /// \param threshold See the link in the description for details
    /// \return Pair containing two vectors. The first one has size k*n and contains calculated coordinates of cluster
    /// centroids.
    /// IMPORTANT NOTE! Its first k elements are equal to the first coordinate of calculated centroids, next k elements
    /// correspond to the second coordinate of calculated centroids, etc. So the order is same as h_input_data's order.
    /// The second element of pair is vector of size N and contains data about input points assignments to clusters,
    /// i.e. i-th element is index of cluster to which i-th input point is assigned to.
    template <class InputIterator>
    pair<thrust::host_vector<double>, thrust::host_vector<int>> Compute(
            const InputIterator h_input_data,
            const int N,
            const int n,
            const int k,
            const double threshold)
    {
        switch (n)
        {
            TEMPLATE_COMPUTE_CASE(1)
            TEMPLATE_COMPUTE_CASE(2)
            TEMPLATE_COMPUTE_CASE(3)
            TEMPLATE_COMPUTE_CASE(4)
            TEMPLATE_COMPUTE_CASE(5)
            TEMPLATE_COMPUTE_CASE(6)
            TEMPLATE_COMPUTE_CASE(7)
            TEMPLATE_COMPUTE_CASE(8)
            TEMPLATE_COMPUTE_CASE(9)
            TEMPLATE_COMPUTE_CASE(10)
            TEMPLATE_COMPUTE_CASE(11)
            TEMPLATE_COMPUTE_CASE(12)
            TEMPLATE_COMPUTE_CASE(13)
            TEMPLATE_COMPUTE_CASE(14)
            TEMPLATE_COMPUTE_CASE(15)
            default:
                throw invalid_argument("n must be at least one and not greater than MAX_n");
        }
    }

}

#endif