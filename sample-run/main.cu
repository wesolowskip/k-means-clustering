#include <cstdlib>
#include <iostream>
#include <ctime>
#include <utility>  //std::pair
#include <cmath>
#include <thrust/host_vector.h>
#include <chrono>

#include "k_means_cpu.hpp"
#include "k_means_gpu_1.cuh"
#include "k_means_gpu_2.cuh"

using namespace std;

/// Creates sample data for k-means clustering functions.
/// \param N Set by the function
/// \param n Set by the function
/// \param k Set by the function
/// \param threshold Set by the function
/// \return Pair of host_vectors, its first item is for cpu k-means clustering algorithm while the second is for
/// gpu implementations
pair<thrust::host_vector<double>, thrust::host_vector<double>> generate_sample_input(int &N, int &n, int &k,
        double &threshold);

class DoubleComparer;

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
int main()
{
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    cout << "Seed for srand: " << seed << endl;
    int N, n, k;
    double threshold;
    auto input_pair = generate_sample_input(N, n, k, threshold);
    cout << "\nN=" << N << " , n=" << n << " , k=" << k << " , threshold=" << threshold << endl;
    pair<thrust::host_vector<double>, thrust::host_vector<int>> cpu_result, gpu_result1, gpu_result2;
    cout << "\nk_means_gpu_1::Compute:\n";
    START_STOPWATCH
    gpu_result1 = k_means_gpu_1::Compute(input_pair.second.begin(), N, n, k, threshold);
    STOP_STOPWATCH
    cout << "\nk_means_gpu_2::Compute:\n";
    START_STOPWATCH
    gpu_result2 = k_means_gpu_2::Compute(input_pair.second.begin(), N, n, k, threshold);
    STOP_STOPWATCH
    cout << "\nk_means_cpu::Compute:\n";
    START_STOPWATCH
    cpu_result = k_means_cpu::Compute(input_pair.first.begin(), N, n, k, threshold);
    STOP_STOPWATCH
    cout << endl;
	cout << boolalpha << "Cpu result with Gpu1 comparison: " << compare_cpu_gpu_results(cpu_result, gpu_result1, n, k)
	    << endl;
    cout << boolalpha << "Gpu1 result with Gpu2 result comparison: " << compare_gpu_results(gpu_result1, gpu_result2)
        << endl;
    return 0;
}

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

pair<thrust::host_vector<double>, thrust::host_vector<double>> generate_sample_input(int &N, int &n, int &k,
                                                                                     double &threshold)
{
    N = 5000000;
    n = 3;
    k = 80;
    threshold = 1e-6;
//    cin >> N >> n >> k >> threshold;
    thrust::host_vector<double> cpu_arg = thrust::host_vector<double>(N*n);
    thrust::host_vector<double> gpu_arg = thrust::host_vector<double>(N*n);
    for (int i = 0; i < N; i++)
    {
        //i-th point
        for (int j = 0; j < n; j++)
        {
            //j-th coordinate
            double coordinate = -600 + 400 * (rand() % 4);   //-600, -200, 200 lub 600
            coordinate += rand() % 101 - 50;
            //Pay attention to the order of items in both vectors
            cpu_arg[i*n + j] = coordinate;
            gpu_arg[j*N + i] = coordinate;
        }
    }
    return std::make_pair(cpu_arg, gpu_arg);
}

bool compare_cpu_gpu_results(const pair<thrust::host_vector<double>, thrust::host_vector<int>>& cpu_result,
                             const pair<thrust::host_vector<double>, thrust::host_vector<int>>& gpu_result,
                             const int n, const int k)
{
    DoubleComparer comparer;
    auto cpu_centroids = cpu_result.first;
    auto gpu_centroids = gpu_result.first;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            if (!comparer(cpu_centroids[i*n + j], gpu_centroids[i + j * k]))
                return false;
    auto cpu_assignments = cpu_result.second;
    auto gpu_assignments = gpu_result.second;
    return thrust::equal(cpu_assignments.begin(), cpu_assignments.end(), gpu_assignments.begin());
}

bool compare_gpu_results(const pair<thrust::host_vector<double>, thrust::host_vector<int>>& gpu_result_1,
                         const pair<thrust::host_vector<double>, thrust::host_vector<int>>& gpu_result_2)
{
    DoubleComparer comparer;
    return gpu_result_1.first.size() == gpu_result_2.first.size() &&
        gpu_result_1.second.size() == gpu_result_2.second.size() &&
        thrust::equal(gpu_result_1.first.begin(), gpu_result_1.first.end(), gpu_result_2.first.begin(), comparer) &&
        thrust::equal(gpu_result_1.second.begin(), gpu_result_1.second.end(), gpu_result_2.second.begin(), comparer);
}