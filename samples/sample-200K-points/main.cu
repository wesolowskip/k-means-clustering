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

#include "utils.h"

using namespace std;


int main()
{
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    cout << "Seed for srand: " << seed << endl;
    const int N = 200000, n = 3, k = 60;
    const double threshold = 1e-4;
    auto input_pair = generate_sample_input(N, n, k);
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
