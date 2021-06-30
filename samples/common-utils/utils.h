#ifndef KMEANSCLUSTERING_UTILS_H
#define KMEANSCLUSTERING_UTILS_H

#include <thrust/host_vector.h>
#include <utility>

using namespace std;

/// Creates samples data for k-means clustering functions.
/// \param N Number of points
/// \param n Dimension
/// \param k Requested number of clusters
/// \return Pair of host_vectors, its first item is for cpu k-means clustering algorithm while the second is for
/// gpu implementations
pair<thrust::host_vector<double>, thrust::host_vector<double>> generate_sample_input(const int N,
                                                                                     int n,
                                                                                     int k);

class DoubleComparer
{
    static constexpr double precision = 1e-6;
public:
    __host__ __device__
    bool operator()(double x, double y)
    {
        return abs(x - y) < precision;
    }
};

bool compare_cpu_gpu_results(const pair<thrust::host_vector<double>, thrust::host_vector<int>>& cpu_result,
                             const pair<thrust::host_vector<double>, thrust::host_vector<int>>& gpu_result,
                             const int n, const int k);

bool compare_gpu_results(const pair<thrust::host_vector<double>, thrust::host_vector<int>>& gpu_result_1,
                         const pair<thrust::host_vector<double>, thrust::host_vector<int>>& result2);

#define START_STOPWATCH {\
                        auto begin = chrono::high_resolution_clock::now();
#define STOP_STOPWATCH auto end = chrono::high_resolution_clock::now();\
                        auto dur = end - begin;\
                        auto ms = chrono::duration_cast<chrono::milliseconds>(dur).count();\
                        auto us = chrono::duration_cast<chrono::microseconds>(dur).count() - ms*1000;\
                        cout << "Measured time: " << ms << "ms" << " " << us << "us\n";    \
                        }

#endif //KMEANSCLUSTERING_UTILS_H
