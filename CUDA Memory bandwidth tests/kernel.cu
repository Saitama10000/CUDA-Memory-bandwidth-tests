#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.h"
#include <cuda.h>
#include <time.h>
#include <stdio.h>

extern "C" __global__ void getClock(int* timings)
{
	int tid = threadIdx.x;
	unsigned long long start = clock64();

	unsigned long long end = clock64();
	timings[tid] = start;
	timings[tid + 1] = end;
}

int main()
{
	const long long size = 1ll << 28;
	int* d_input;
	int* d_output;
	cudaErr(cudaMalloc((void**)&d_input, size * sizeof(int)));
	cudaErr(cudaMalloc((void**)&d_output, size * sizeof(int)));


	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction get_clock;
	CUfunction get_time;

	cuErr(cuInit(0));
	cuErr(cuDeviceGet(&device, 0));
	cuErr(cuCtxCreate(&context, 0, device));
	cuErr(cuModuleLoad(&module, "kernel.ptx"));
	cuErr(cuModuleGetFunction(&get_clock, module, "get_clock"));
	cuErr(cuModuleGetFunction(&get_time, module, "get_time"));

	unsigned long long h_clock[2];
	CUdeviceptr d_clock;
	cuErr(cuMemAlloc(&d_clock, 2 * sizeof(h_clock[0])));

	for (int i = 0; i < 16; i++)
	{
		{
			void* args[] = { &d_clock };
			cuErr(cuLaunchKernel(get_clock, 1, 1, 1, 1, 1, 1, 0, NULL, (void**)args, NULL));
			cuErr(cuMemcpyDtoH((void*)&h_clock, d_clock, 2 * sizeof(h_clock[0])));
			printf("Clocks: %lld %lld %lld\n", h_clock[0], h_clock[1], h_clock[1] - h_clock[0]);
		}
		{
			void* args[] = { &d_clock };
			cuErr(cuLaunchKernel(get_time, 1, 1, 1, 1, 1, 1, 0, NULL, (void**)args, NULL));
			cuErr(cuMemcpyDtoH((void*)&h_clock, d_clock, 2 * sizeof(h_clock[0])));
			printf("Time: %lld %lld %lld\n", h_clock[0], h_clock[1], h_clock[1] - h_clock[0]);
		}
		{
			getClock<<<1, 1>>>((void*)d_clock);
			cudaErr(cudaMemcpy((void*)&h_clock, (void*)d_clock, 2 * sizeof(h_clock[0]), cudaMemcpyDeviceToHost));
			printf("Time: %lld %lld %lld\n", h_clock[0], h_clock[1], h_clock[1] - h_clock[0]);
		}
	}



	cudaErr(cudaFree(d_input));
	cudaErr(cudaFree(d_output));
	cudaDeviceReset();
	return 0;
}