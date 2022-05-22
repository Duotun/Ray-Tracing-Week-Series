
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <algorithm>   //well, cuda seems doesn't support c++ 17 but we could add the command line
#include <iostream>
#include <time.h>  //for clock timing
#include <curand_kernel.h>   //for random numbers

//check cudaError_t with a macro to output to stdout
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result!=cudaSuccess) {  //cudaSuccess equals to 0
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
	}
}

__device__  __host__ //HOST INDICATES running on GPU, while __host__ indicates CPU 
uint32_t ConvertColor(float c_val)   //also consider the gamma correction, put this conversion into GPU
{
    //convert to one byte
    float gamma = 2.0f;
    c_val = std::fabs(std::pow(c_val, 1.0f / gamma));   //save extra declaration
    c_val = std::clamp(255.999f * c_val, 0.0f, 255.0f);  // c++ 17 support for clamping
    //c_val = (255.999f * c_val) > 255.0f ? 255.0f : 255.999f * c_val;

    return static_cast<uint32_t>(c_val);
}


//init randon number for each pixels and hence each thread
__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

