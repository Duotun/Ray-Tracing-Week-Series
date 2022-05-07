
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
#include "hitable.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include "hit_record.hpp"
#include "material.hpp"

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

    ray cur_ray(r.org, r.dir);
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
    hitable** world, curandState * rand_state, camera **cam, int m_depth)   
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

        ray r = (*cam)->get_ray(u, v, &local_rand_state); r.m_depth = m_depth;
        col += Rendercolor(r, world, &local_rand_state);
    }
    
    //write rgb three channels using Vector3
    rand_state[pixel_index] = local_rand_state;
    fb[pixel_index] = col / float(ns);
}

#define RND (curand_uniform(&local_rand_state))
__global__ void create_random_scene(hitable** d_list, hitable** d_world, camera** d_cam, int nx, int ny, curandState* rand_state)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    curandState local_rand_state = *rand_state;
    d_list[0] = new sphere(Vector3(0, -1000.0, -1), 1000,
        new lambertian(Vector3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            Vector3 center(a + RND, 0.2, b + RND);
            if (choose_mat < 0.8f) {
                d_list[i++] = new sphere(center, 0.2,
                    new lambertian(Vector3(RND * RND, RND * RND, RND * RND)));
            }
            else if (choose_mat < 0.95f) {
                d_list[i++] = new sphere(center, 0.2,
                    new metal(Vector3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
            }
            else {
                d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
            }
        }
    }
    d_list[i++] = new sphere(Vector3(0, 1, 0), 1.0, new dielectric(1.5));
    d_list[i++] = new sphere(Vector3(-4, 1, 0), 1.0, new lambertian(Vector3(0.4, 0.2, 0.1)));
    d_list[i++] = new sphere(Vector3(4, 1, 0), 1.0, new metal(Vector3(0.7, 0.6, 0.5), 0.0));
    *rand_state = local_rand_state;
    *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

    Vector3 lookfrom(13, 2, 3);
    Vector3 lookat(0, 0, 0);
    float dist_to_focus = 10.0; (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_cam = new camera(lookfrom,
        lookat,
        Vector3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

//kernerl for creating worlds
__global__ void Create_World(hitable** d_list, hitable** d_world, camera** d_cam, int nx, int ny, curandState * rand_state)
{
    //only execute the construction of the world once
   
    /*if (threadIdx.x == 0 && blockIdx.x == 0)   //defocus scene
    {
        d_list[0] = new sphere(Vector3(0, 0, -1), 0.5,
            new lambertian(Vector3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(Vector3(0, -100.5, -1), 100,
            new lambertian(Vector3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(Vector3(1, 0, -1), 0.5,
            new metal(Vector3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(Vector3(-1, 0, -1), 0.5,
            new dielectric(1.5));
        d_list[4] = new sphere(Vector3(-1, 0, -1), -0.45,
            new dielectric(1.5));

        Vector3 lookfrom(3, 3, 2);
        Vector3 lookat(0, 0, -1);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 2.0;

        *d_world = new hitable_list(d_list, 5);
        *d_cam = new camera(lookfrom,
                            lookat,
                            Vector3(0, 1, 0),
                            20.0,
                            float(nx)/float(ny),
                            aperture,
                            dist_to_focus);
    }*/

}

//kernel for deleteing worlds
__global__ void Free_World(hitable** d_list, hitable** d_world, camera** d_camera)
{
    for (int i = 0; i < 5; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

__global__ void free_random_scene(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

int main()
{
    //set up the image
    uint32_t  width = 1200;
    uint32_t  height = 600;
    uint32_t  ns = 150;
    uint32_t  m_depth = 50;   //maximum tracing depth
    uint32_t  num_pixels = width * height;
    size_t fb_size = (size_t)num_pixels * sizeof(Vector3);

    //allocate fram buffer (FB)
     Vector3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));


    //allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));
    rand_init << <1, 1 >> > (d_rand_state2);  //for random_scene, one init is fine

    // make the world of hitables
    hitable** d_list;   //d_ for the device only data
    int num_hittables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables*sizeof(hitable* )));   //prepare to create two hitable objects
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**) & d_world, sizeof(hitable*)));
    camera** d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));  // we may need to change camera* , so **
    create_random_scene << <1, 1 >> > (d_list, d_world, d_cam, width, height, d_rand_state2);
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
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //very simple projected plane calculation
    render<<<blocks, threads >>>(fb, width, height, ns,
        d_world,
        d_rand_state,
        d_cam,
        m_depth);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    double time_seconds = ((double)(stop - start) / CLOCKS_PER_SEC);
    std::cerr << "took " << time_seconds << " seconds.\n";


    //output FB as PPM Image
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = int(height - 1); j >= 0; j--) {
        for (int i = 0; i < int(width); i++) {
            size_t pixel_index = j * width + i;
            auto r = ConvertColor(fb[pixel_index].x());
            auto g = ConvertColor(fb[pixel_index].y());
            auto b = ConvertColor(fb[pixel_index].z());
         
            std::cout << r << " " << g << " " << b << "\n";
        }
    }

    //cleanup
    free_random_scene<< <1, 1 >> > (d_list, d_world, d_cam);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));

    //last check for cuda memory
    cudaDeviceReset();
}