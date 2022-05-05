#pragma once
#ifndef RAY_H
#define RAY_H
#include "vector.hpp"
#include <limits>

class ray
{
public:
	__device__ ray() {
		m_tmin = 0.0001f; m_tmax = std::numeric_limits<float>::max();}

	__device__ ~ray() = default;
	__device__ ray(const point3 & origin, const Vector3 & direction):
		org(origin), dir(unit_vector(direction)), m_tmin(0.001), m_tmax(std::numeric_limits<float>::max()) {}

	__device__ ray(point3& origin, Vector3& direction) :
		org(origin), dir(unit_vector(direction)), m_tmin(0.001), m_tmax(std::numeric_limits<float>::max()) {}

	point3 org;
	Vector3 dir;
	int m_depth = 3;  //at least or the worst case , perform the recursion 3 times
	float m_tmin, m_tmax;    // also set min and max segmentations in the ray class
	__device__ void set_min(float vmin)
	{
		m_tmin = vmin;
	}

	__device__ void set_max(float vmax)
	{
		m_tmax = vmax;
	}

	__device__ point3 at(float t)
	{
		dir.normalize();
		return org + t * dir;
	}

};

#endif 