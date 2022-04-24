#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>   //for provide threadIdx, blockDim, etc.

//function to add the elements of two arrays
__global__   //indicate the kernel function
void add(int n, float* x, float* y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = 0; i < n; i+=stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	int N = 1 << 20; // 1M elements
	
	//prepare the data, Allocate unified memory both in CPU and GPU
	float* x, * y;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	//initialize x and y arrays on the host
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	//run the kernel either on GPU with kernerls
	add << <1, 256 >> > (N, x, y);

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//Check for errors (all values should be 3.0f)
	float maxVal = 0.0f;
	for (int i = 0; i < N; i++)
		maxVal = std::fmax(maxVal, std::fabs(y[i] - 3.0f));
	std::cout << "Max Error: " << maxVal << std::endl;

	//Free Memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}