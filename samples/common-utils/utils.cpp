#include "utils.h"

pair<thrust::host_vector<double>, thrust::host_vector<double>> generate_sample_input(const int N,
                                                                                     const int n,
                                                                                     const int k)
{
    thrust::host_vector<double> cpu_arg = thrust::host_vector<double>(N*n);
    thrust::host_vector<double> gpu_arg = thrust::host_vector<double>(N*n);
    for (int i = 0; i < N; i++)
    {
        //i-th point
        for (int j = 0; j < n; j++)
        {
            //j-th coordinate
            double coordinate = -600 + 400 * (rand() % 4);   //-600, -200, 200 or 600
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