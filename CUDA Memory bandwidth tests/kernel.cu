#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.h"
#include "kernel.ptx"
#include <cuda.h>
#include <time.h>
#include <stdio.h>

int main()
{
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction get_clock;
	CUfunction get_time;

	cuErr(cuInit(0));
	cuErr(cuDeviceGet(&device, 0));
	cuErr(cuCtxCreate(&context, 0, device));
	cuErr(cuModuleLoadData(&module, kernel_ptx));
	cuErr(cuModuleGetFunction(&get_clock, module, "get_clock"));
	cuErr(cuModuleGetFunction(&get_time, module, "get_time"));

	unsigned long long h_clock[1];
	unsigned long long clock = 0;
	unsigned long long time = 0;
	unsigned long long n = 0;
	
	CUdeviceptr d_clock;
	cuErr(cuMemAlloc(&d_clock, sizeof(h_clock[0])));
	
	while(true)
	{
		{
			void* args[] = { &d_clock };
			cuErr(cuLaunchKernel(get_clock, 1, 1, 1, 1, 1, 1, 0, NULL, (void**)args, NULL));
			cuErr(cuMemcpyDtoH((void*)&h_clock, d_clock, sizeof(h_clock[0])));
			clock += h_clock[0];
		}
		{
			void* args[] = { &d_clock };
			cuErr(cuLaunchKernel(get_time, 1, 1, 1, 1, 1, 1, 0, NULL, (void**)args, NULL));
			cuErr(cuMemcpyDtoH((void*)&h_clock, d_clock, sizeof(h_clock[0])));
			time += h_clock[0];
		}
		n += 1;
		printf("\rClock: %8.2f Time: %8.2f", (double)(clock) / n, (double)(time) / n);
	}

	cudaErr(cudaDeviceReset());
	return 0;
}