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
#pragma endregion

//Using
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

//Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

//Utility Functions
inline double degrees_to_radians(double degrees)
{
	return degrees * pi / 180.0;
}

//generate random number (0 <= r < 1)
inline double random_double() {
	// Returns a random real in [0, 1).
	return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
	// Returns a random real in [min, max).
	return  min + (max - min) * random_double();
}

inline double random_double_generatorway()
{
	static std::uniform_real_distribution<double> disctribution(0.0, 1.0);
	static std::mt19937 generator;
	return disctribution(generator);
}

//common Headers
#include "ray.hpp"
#include "vector.hpp"


#endif // !UTILITY_H

