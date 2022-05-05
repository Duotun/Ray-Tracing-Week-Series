#pragma once
#ifndef UTILITY_H
#define UTILITY_H

#pragma region
//includes
#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <random>
#include "device_launch_parameters.h"
#include <curand_kernel.h>   //used for the cuda random generator methods
#include "vector.hpp"

#pragma endregion

//Using
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

//Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

//Utility Functions
__host__ __device__ inline double degrees_to_radians(double degrees)
{
	return degrees * pi / 180.0;
}

//generate random number (0 <= r < 1)
__host__  inline double random_double() {
	// Returns a random real in [0, 1).
	return rand() / (RAND_MAX + 1.0);
}

__host__  inline double random_double(double min, double max) {
	// Returns a random real in [min, max).
	return  min + (max - min) * random_double();
}

__host__ inline double random_double_generatorway()
{
	static std::uniform_real_distribution<double> disctribution(0.0, 1.0);
	static std::mt19937 generator;
	return disctribution(generator);
}

//let's write the random methods suitable for the __device__
__device__ float random_curand(curandState* rand_state)
{
	return curand_uniform(rand_state);

}

__device__ float random_curand_range(curandState *rand_state, float min, float max)
{
	return   min+(max-min)* curand_uniform(rand_state);
}


#endif // !UTILITY_H

