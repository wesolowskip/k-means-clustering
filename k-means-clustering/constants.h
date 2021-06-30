#ifndef KMEANSCLUSTERING__CONSTANTS_H
#define KMEANSCLUSTERING__CONSTANTS_H

const int MAX_n = 15;
const int MAX_ITERS = 1000000;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line) 
{
	if (err != cudaSuccess) 
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define TEMPLATE_COMPUTE_CASE( i ) case i: return templated_compute< i >(h_input_data, N, k, threshold);

#endif