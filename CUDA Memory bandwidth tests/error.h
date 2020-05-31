#pragma once
#include <stdio.h>
#include <cuda.h>

#define cudaErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t err, const char* file, int line)
{
	if (err)
	{
		fprintf(stderr, "cudaError: %s\n\tfile<\"%s\">\n\t\tline:%d \n", cudaGetErrorString(err), file, line);
		exit(err);
	}
}

#define cuErr(err)  { gpuAssert((err), __FILE__, __LINE__); }
inline void gpuAssert(CUresult err, const char* file, const int line)
{
	if (err)
	{
		fprintf(stderr, "cuError: %s\n\tfile<\"%s\">\n\t\tline:%d \n", cudaGetErrorString((cudaError_t)err), file, line);
		exit(err);
	}
}