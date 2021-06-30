#ifndef KMEANSCLUSTERING_K_MEANS_CPU_HPP
#define KMEANSCLUSTERING_K_MEANS_CPU_HPP

#include <utility>
#include <thrust/host_vector.h>
#include <limits> //DBL_MAX
#include "constants.h"

using namespace std;

namespace k_means_cpu
{
	template <class InputIterator>
	double calculate_squared_distance(
		const InputIterator points,
		const thrust::host_vector<double>& centroids,
		const int which_point,
		const int which_centroid,
		const int n)
	{
		double squared_distance = 0;
		for (int i = 0; i < n; i++)
		{
			double tmp = points[which_point * n + i] - centroids[which_centroid * n + i];
			squared_distance += tmp * tmp;
		}
		return squared_distance;
	}

	/// Basic sequential implementation of k-means clustering algorithm using CPU.
	/// Implementation based on http://users.eecs.northwestern.edu/~wkliao/Kmeans/index.html.
	/// \tparam RandomAccessIterator input_data type
	/// \param input_data Iterator to double coordinates of data points. Corresponding collection should be
    /// of size N*n and first n elements are coordinates of the first data point, next n elements of the second one, etc.
	/// \param N Number of data points
    /// \param n Dimension of data points, must be less or equal to MAX_n defined in constants header
	/// \param k Requested number of clusters
	/// \param threshold See the link in the description for details
	/// \return Pair containing two vectors. The first one has size k*n and contains calculated coordinates of cluster
	/// centroids. Its first n elements are coordinates of the first centroid, next n elements of the second one etc.
	/// The second element of pair is vector of size N and contains data about input points assignments to clusters,
	/// i.e. i-th element is index of cluster to which i-th input point is assigned to.
	template <class RandomAccessIterator>
	pair<thrust::host_vector<double>, thrust::host_vector<int>> Compute(
		const RandomAccessIterator input_data,
		const int N,
		const int n,
		const int k,
		const double threshold)
	{
	    if (k > N)
	        throw invalid_argument("k must be less or equal to N");
		thrust::host_vector<double> centroids(k * n);
		thrust::host_vector<double> new_centroids(k * n);
		thrust::host_vector<int> new_centroids_size(k);
		thrust::copy(input_data, input_data + k*n, centroids.begin());
		thrust::host_vector<int> closest_centroid_indices(N, -1);
		int delta;
		int iter = 0;
		do
		{
			delta = 0;
			thrust::fill(new_centroids.begin(), new_centroids.end(), 0);
			thrust::fill(new_centroids_size.begin(), new_centroids_size.end(), 0);
			for (int i = 0; i < N; i++)
			{
				int closest_centroid_index = -1;
				double min_squared_distance = DBL_MAX;
				for (int j = 0; j < k; j++)
				{
					double squared_distance = calculate_squared_distance(input_data, centroids, i, j, n);
					if (squared_distance < min_squared_distance)
					{
						min_squared_distance = squared_distance;
						closest_centroid_index = j;
					}
				}
				if (closest_centroid_indices[i] != closest_centroid_index)
				{
					delta++;
					closest_centroid_indices[i] = closest_centroid_index;
				}
				for (int j = 0; j < n; j++)
					new_centroids[closest_centroid_index*n + j] += input_data[i*n + j];
				new_centroids_size[closest_centroid_index] ++;
			}
			for (int i = 0; i < k; i++)
			{
				int size = new_centroids_size[i];
				if (size)
				{
					for (int j = 0; j < n; j++)
						centroids[i*n + j] = new_centroids[i*n + j] / (double)size;
				}
			}
		} while ((double)delta / N > threshold && (iter++ < MAX_ITERS));
		if (iter >= MAX_ITERS)
		    printf("Max iteration count reached\n");
		return std::make_pair(centroids, closest_centroid_indices);
	}
}

#endif