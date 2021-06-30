#ifndef KMEANSCLUSTERING_K_MEANS_GPU_1_CUH
#define KMEANSCLUSTERING_K_MEANS_GPU_1_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/discard_iterator.h>
#include "common_kernels.cuh"
#include "constants.h"


using namespace std;
using namespace common_kernels;

namespace k_means_gpu_1
{
	///A useful functor for matrix row reduction (reducing rows of matrix to single column)
	class matrix_row_index : public thrust::unary_function<int, int>
	{
		const int row_size;
	public:
		matrix_row_index(int row_size) :row_size{ row_size } {}
		/// Assuming that matrix is stored in single, consistent region of memory in which first row_size elements
		/// are items from first row, next row_size elements are items from second row etc. this function converts
		/// memory element index to row index.
		/// \param Index of element in consistent memory (array, vector)
		/// \return Index of row of matrix
		__host__ __device__ int operator()(int index) const
		{
			return index / row_size;
		}
	};

    /// For details see https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    /// \tparam T
    /// \param sdata
    /// \param tid
	template <class T>
	__device__ void warp_reduce(volatile T* sdata, int tid)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	/// Primary reduction kernel, for reduction details see:
	/// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
    /// \tparam n Dimension of points
	/// \param points Thrust pointer to points coordinates. Order of items is same as order of h_input_data
	/// parameter of k_means_gpu_1::Compute function.
	/// \param closest_centroid_indices i-th element is equal to index of closest centroid for i-th data point
	/// \param output_sums Result vector. It contains block-level reduced sums of coordinates of points assigned to
	/// each centroid in specific order. Structure of this vector is as follows: it can be split into n consecutive parts,
	/// each of k*numBlocks elements. Every such part describes reduces sums for specific coordinate. Let us consider
	/// i-th part describing i-th coordinate. We can split such part into k parts, each describing specific centroid
	/// and contains numBlocks subsequent items. Considering j-th such part describing j-th centroid, it contains
	/// block-level reduced sums for i-th coordinate and j-th centroid and its l-th element describes reduced sum in
	/// l-th block. Note that we can treat output_sums vector as a matrix in which each row has numBlocks element.
	/// Summing values in each row of such matrix would give us a vector of a form:
	/// [s_1_1, s_1_2, ..., s_1_k, s_2_1, ..., s_2_k, ..., s_n_1, ... s_n_k]
	/// where s_i_j is sum of i-th coordinate of points assigned to j-th centroid
	/// \param output_counts Result vector. It contains reduced numbers of points assigned to centroids for every block.
	/// Its size is k*numBlocks. Structure of this vector is as follows: it can be split into consecutive parts of equal
	/// size and i-th part describes reduces counts for i-th centroids. Every such part has numBlocks items and specific
	/// element is equal to reduced number of points belonging to i-th centroid considered in specific block.
	/// Note that treating output_counts vectors as a matrix, where each row has numBlocks items we can reduce (sum)
	/// rows of such matrix and as a result we will get a complete vector in which i-th element will be equal to number
	/// of points assigned to i-th centroids
	/// \param N Number of input points
	/// \param k Requested number of clusters
    template <int n>
    __global__ void calculate_new_centroids(
            const thrust::device_ptr<double> points,
            const thrust::device_ptr<int> closest_centroid_indices,
            thrust::device_ptr<double> output_sums,
            thrust::device_ptr<int> output_counts,
            const int N,
            const int k)
    {
        extern __shared__ double sdata[];
        int *sdata_int = (int *)&sdata[0];

        const int tid = threadIdx.x;
        const int idx = blockIdx.x * blockDim.x * 2 + tid;
        const int gridSize = blockDim.x * 2 * gridDim.x;

        int loop_idx, loop_strided_idx;
        for (int i = 0; i < k; i++)
            //Reducing number of assigned points and coordinates sum for i-th centroid
        {
            //Firstly we reduce number of points assigned to i-th centroid
            int tmp_count = 0;
            loop_idx = idx;
            //while loop based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf - Chapter
            //'Multiple Adds/Thread'
            while (loop_idx < N)
            {
                //If loop_idx-th point has been assigned to i-th centroid, we increment number of assigned points
                if (closest_centroid_indices[loop_idx] == i)
                    tmp_count++;
                loop_strided_idx = loop_idx + blockDim.x;
                //Like above
                if (loop_strided_idx < N && closest_centroid_indices[loop_strided_idx] == i)
                    tmp_count++;
                loop_idx += gridSize;
            }
            //It is very important that for threads outside of range (when N is not 1024 multiply) tmp_count is 0
            sdata_int[tid] = tmp_count;
            __syncthreads();
            for (int s = (blockDim.x >> 1); s > 32; s >>= 1)
            {
                if (tid < s)
                    sdata_int[tid] += sdata_int[tid + s];
                __syncthreads();
            }
            if (tid < 32)   //see https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
                warp_reduce(sdata_int, tid);
            //Now we have calculated how many points are assigned to i-th centroid in current block
            if (tid == 0)
                output_counts[i * gridDim.x + blockIdx.x] = sdata_int[0];
            __syncthreads();
            //Now it's time for reducing sums for every coordinate
            for (int j = 0; j < n; j++)  //j is coordinate index
            {
                //Reduction like above
                double tmp_sum = 0.;
                loop_idx = idx;
                while (loop_idx < N)
                {
                    //If loop_idx-th point has been assigned to i-th centroid, we add j-th coordinate of point to local
                    //sum of j-th coordinates
                    if (closest_centroid_indices[loop_idx] == i)
                        tmp_sum += points[N*j + loop_idx];
                    loop_strided_idx = loop_idx + blockDim.x;
                    //Like above
                    if (loop_strided_idx < N && closest_centroid_indices[loop_strided_idx] == i)
                        tmp_sum += points[N*j + loop_strided_idx];
                    loop_idx += gridSize;
                }
                sdata[tid] = tmp_sum;
                __syncthreads();
                for (int s = (blockDim.x >> 1); s > 32; s >>= 1)
                {
                    if (tid < s)
                        sdata[tid] += sdata[tid + s];
                    __syncthreads();
                }
                if (tid < 32)
                    warp_reduce(sdata, tid);
                if (tid == 0)
                    output_sums[j * k * gridDim.x + gridDim.x * i + blockIdx.x] = sdata[0];
                __syncthreads();
            }

        }

    }

    /// Given sums of every coordinate and numbers of points assigned to each centroid, this function calculates
    /// new coordinates of centroids (calculates corresponding arithmetic means)
    /// \tparam n
    /// \param sums
    /// \param counts
    /// \param centroids
    /// \param k
    template <int n>
    __global__ void average_centroids(const thrust::device_ptr<double> sums,
                                      const thrust::device_ptr<int> counts,
                                      thrust::device_ptr<double> centroids,
                                      const int k
    )
    {
        //Grid-stride-loop
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < k; i += stride)
        {
            int count = counts[i%k];
            if (count) //if count == 0 then we do not change centroid coordinates
            {
                for (int j=0, coordinate_index = i; j < n; j++, coordinate_index += k)
                    centroids[coordinate_index] = sums[coordinate_index] / (double)count;
            }
        }
    }

    /// Templated implementation of first k-means clustering algorithm using GPU.
    /// This implementation is based particularly on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
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
            const double threshold)
    {
        if (k > N)
            throw invalid_argument("k must be less or equal to N");

        const int blockSize = 1024;
        const int numBlocks_N_threads = (N + blockSize - 1) / blockSize;
        const int numBlocks_k_threads = (k + blockSize - 1) / blockSize;

        int devId;
        HANDLE_ERROR(cudaGetDevice(&devId));
        int numSMs;
        HANDLE_ERROR(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
        const int numBlocks_reduce = numSMs * 32;
        //An alternative would be:
        //const int numBlocks_reduce = (N + 2*blockSize - 1) / (2*blockSize);
        //but then it is necessary for performance to remove 'while' loops that initialize
        //shared memory in 'calculate_new_centroids' function

        thrust::device_vector<double> input_data(h_input_data, h_input_data + N * n);
        thrust::device_vector<double> centroids(k*n);
        thrust::device_vector<int> closest_centroid_indices(N, -1);
        thrust::device_vector<int> prev_closest_centroid_indices(N);
        thrust::device_vector<double> reduced_sums(n*k*numBlocks_reduce);
        thrust::device_vector<int> reduced_counts(k*numBlocks_reduce);
        thrust::device_vector<int> counts(k);
        thrust::device_vector<double> sums(n*k);

        initialize_centroids<n> <<<numBlocks_k_threads, blockSize >>> (input_data.data(), centroids.data(), N, k);

        int delta;
        int iter = 0;
        do
        {
            closest_centroid_indices.swap(prev_closest_centroid_indices);
            find_closest_centroids<n> <<<numBlocks_N_threads, blockSize >>> (input_data.data(), centroids.data(),
                                                                             closest_centroid_indices.data(), N, k);
            //Calculating number of points which assignment to cluster has changed
            delta = thrust::inner_product(prev_closest_centroid_indices.begin(),
                                          prev_closest_centroid_indices.end(),
                                          closest_centroid_indices.begin(),
                                          0,
                                          thrust::plus<int>(),
                                          thrust::not_equal_to<int>())
                    ;
            calculate_new_centroids <n> <<<numBlocks_reduce, blockSize, blockSize * sizeof(double) >>> (
                    input_data.data(),
                    closest_centroid_indices.data(),
                    reduced_sums.data(),
                    reduced_counts.data(),
                    N, k
            );
            //Now we have reduced sums of coordinates and numbers of points assigned to each centroids on block level.
            //Moreover, reduced items are in such order that we can perform matrix-like row reduction (calculate
            //sums of each row). Every row of matrix has numBlocks_reduce items
            auto it = thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    matrix_row_index(numBlocks_reduce)
            );
            thrust::reduce_by_key(
                    it,
                    it + reduced_sums.size(),
                    reduced_sums.begin(),
                    thrust::make_discard_iterator(),
                    sums.begin()
            );
            thrust::reduce_by_key(
                    it,
                    it + reduced_counts.size(),
                    reduced_counts.begin(),
                    thrust::make_discard_iterator(),
                    counts.begin()
            );
            //Calculating new centroids' coordinates
            average_centroids<n> <<<numBlocks_k_threads, blockSize >>> (
                    sums.data(),
                    counts.data(),
                    centroids.data(),
                    k
            );
        } while ((double)delta / N > threshold && (iter++ < MAX_ITERS));
        if (iter >= MAX_ITERS)
            printf("Max iteration count reached\n");
        //Copying results to host
        thrust::host_vector<double> h_centroids{ centroids };
        thrust::host_vector<int> h_centroid_assignments{ closest_centroid_indices };
        return std::make_pair(h_centroids, h_centroid_assignments);
    }

    /// Wrapper for first implementation of k-means clustering using GPU.
    /// Using the fact that n has its maximum value we can use template with parameter n (you can find similar
    /// example here https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).
    /// \tparam InputIterator h_input_data type
    /// \param h_input_data Iterator to double coordinates of data points. Corresponding collection should be
    /// of size N*n.
    /// IMPORTANT NOTE! Order of items is different from the order of corresponding parameter in k_means_cpu::Compute function.
    /// Here, first N elements are coordinates of first coordinate of all data points. Next N elements correspond to
    /// second coordinate of all data points and so forth
    /// \param N Number of data points
    /// \param n Dimension of data points, must be less or equal to MAX_n defined in constants header
    /// \param k Requested number of clusters
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