#pragma once
//define the ray from the vector3 class
#ifndef RAY_H
#define RAY_H
#include "vector.hpp"
#include <limits>

class ray {

public:
	point3 org;
	Vector3 dir;
	int m_depth = 3; //at least recursive 3....
	mutable double m_tmin, m_tmax;
	double tm;
	//contructor and utlilty methods, here not normalized dir, but we could do that by calling unit_vector()
	ray() : m_tmin(0.0001), m_tmax(infinity) {}
	ray(const point3& origin, const Vector3& direction, double time = 0.0) noexcept :  //assume the ray is shooting forward, so m_tmin is 0.0001 for shadow acne tolerance
		org(origin), dir(unit_vector(direction)), tm(time), m_tmin(0.0001), m_tmax(infinity) {}
	ray(point3& origin, Vector3& direction, double time = 0.0) noexcept :
		org(origin), dir(unit_vector(direction)), tm(time), m_tmin(0.0001), m_tmax(infinity) {
	}  //infinity is greater than max limits
	//make sure dir is normalized
	point3 origin() const { return org; }
	Vector3 direction() const { return dir; }

	
	point3 at(double t) {
		dir.normalize();
		return  org + t * dir;
	}

	void set_min(double vmin) {
		m_tmin = vmin;
	}

	void set_max(double vmax) {
		m_tmax = vmax;
	}
	double time()
	{
		return tm;
	}

	double time() const
	{
		return tm;
	}
};
#endif // ! RAY_H

