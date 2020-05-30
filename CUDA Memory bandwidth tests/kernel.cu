/*
	I know I'm not checking for cuda errors but this is 
	just a small piece of code for a small benchmark.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void read(int* input, int* output, const long long size)
{
	long long gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid > size) return;

	output[gid] = input[gid];
}


int main()
{
	const long long size = 1ll << 28;
	int* d_input;
	int* d_output;
	cudaMalloc((void**)&d_input, size * sizeof(int));
	cudaMalloc((void**)&d_output, size * sizeof(int));

	int blockSize = 256;
	int gridSize = size / blockSize;
	read << <gridSize, blockSize >> > (d_input, d_output, size);

	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}