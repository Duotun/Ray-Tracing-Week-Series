#pragma once
#ifndef UTILITY_H
#define UTILITY_H

#pragma region
//includes
#include <cmath>
#include <limits>
#include <memory>
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

//common Headers
#include "ray.hpp"
#include "vector.hpp"


#endif // !UTILITY_H

