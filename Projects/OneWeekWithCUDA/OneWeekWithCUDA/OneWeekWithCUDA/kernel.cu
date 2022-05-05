
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <algorithm>   //well, cuda seems doesn't support c++ 17 but we could add the command line
#include <iostream>
#include <time.h>  //for clock timing
#include <curand_kernel.h>   //for random numbers

//for the world construction
#include "vector.hpp"
#include "ray.hpp"
#include "hitable_list.hpp"
#include "sphere.hpp"
#include "camera.hpp"

//check cudaError_t with a macro to output to stdout
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
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


__device__     //make it iteration which is more suitable for the CUDA Implementation
Vector3 Rendercolor(ray& r, hitable **world, curandState* local_rand_state)  //assum the direction vectors are normalized
{
    int depth = r.m_depth;  
    Vector3 cur_attenuation = Vector3(1.0f, 1.0f, 1.0f);

    ray cur_ray = r;
    //perform the iteration ray tracing
    for (int i = 0; i < depth; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, rec)) {
            ray scattered;
            Vector3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return Vector3(.0f, 0.0f, 0.0f);
            }
        }
        else {   //if no intersection ends here;
            float t = 0.5f * (cur_ray.dir.y() + 1.0f);
            //printf("t: %d", t);
            return cur_attenuation* ((1.0f - t) * Vector3(1.0f, 1.0f, 1.0f) + t * Vector3(0.5f, 0.7f, 1.0f));
        }
    }
    
    //exceed iterations, ok black totally
    return Vector3(0.0f, 0.0f, 0.0f);
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

__global__ //define the kernels
void render(Vector3* fb, int max_x, int max_y, int ns, 
    hitable** world, curandState * rand_state, camera **cam, int m_depth = 50)   
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;  

    
    //perform the sampling for anti-aliasing here using rand_state
    Vector3 col(0.0f, 0.0f, 0.0f);
    curandState local_rand_state = rand_state[pixel_index];
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);   //curand_uniform wants *curandState
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

        ray r = (*cam)->get_ray(u, v); r.m_depth = m_depth;
        col += Rendercolor(r, world, &local_rand_state);
    }
    
    //write rgb three channels using Vector3
    rand_state[pixel_index] = local_rand_state;
    fb[pixel_index] = col / float(ns);
}


//kernerl for creating worlds
__global__ void Create_World(hitable** d_list, hitable** d_world, camera** d_cam)
{
    //only execute the construction of the world once
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *(d_list) = new sphere(Vector3(0, 0, -1), 0.5f);
        *(d_list+1) = new sphere(Vector3(0, -100.5f,-1.0f), 100.0f);
        *d_world = new hitable_list(d_list, 2);
        *d_cam = new camera();
    }

}

//kernel for deleteing worlds
__global__ void Free_World(hitable** d_list, hitable** d_world, camera** d_camera)
{
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    delete* d_camera;
}


int main()
{
    //set up the image
    uint32_t  width = 1200;
    uint32_t  height = 600;
    uint32_t  ns = 100;
    uint32_t  m_depth = 50;   //maximum tracing depth
    uint32_t  num_pixels = width * height;
    size_t fb_size = (size_t)num_pixels * sizeof(Vector3);

    //allocate fram buffer (FB)
     Vector3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));


    //allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));


    // make the world of hitables
    hitable** d_list;   //d_ for the device only data
    checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hitable* )));   //prepare to create two hitable objects
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**) & d_world, sizeof(hitable*)));
    camera** d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));  // we may need to change camera* , so **
    Create_World << <1, 1 >> > (d_list, d_world, d_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //std::cout << (*d_cam)->lower_left_corner << std::endl;
    //prepare threads allocation
    int tx = 8, ty = 8;     //into blocks of 8x8 threads
    clock_t start, stop;
    start = clock();
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);


    //perform the random init (random number initialization for the renderer)
    render_init << <blocks, threads >> > (width, height, d_rand_state);
    //very simple projected plane calculation
    render<<<blocks, threads >>>(fb, width, height, ns,
        d_world,
        d_rand_state,
        d_cam
        );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    double time_seconds = ((double)(stop - start) / CLOCKS_PER_SEC);
    std::cerr << "took " << time_seconds << " seconds.\n";


    //output FB as PPM Image
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            auto r = ConvertColor(fb[pixel_index].x());
            auto g = ConvertColor(fb[pixel_index].y());
            auto b = ConvertColor(fb[pixel_index].z());
         
            std::cout << r << " " << g << " " << b << "\n";
        }
    }

    //cleanup
    Free_World << <1, 1 >> > (d_list, d_world, d_cam);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));

    //last check for cuda memory
    cudaDeviceReset();
}