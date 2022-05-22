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
	//contructor and utlilty methods, here not normalized dir, but we could do that by calling unit_vector()
	ray(): m_tmin(std::numeric_limits<double>::min()), m_tmax(std::numeric_limits<double>::max()){}
	ray(const point3& origin, const Vector3& direction) noexcept:  //assume the ray is shooting forward, so m_tmin is 0.0001 for shadow acne tolerance
		org(origin), dir(unit_vector(direction)), m_tmin(0.001), m_tmax(std::numeric_limits<double>::max()){}
	ray(point3& origin, Vector3& direction) noexcept :
		org(origin), dir(unit_vector(direction)), m_tmin(0.001), m_tmax(std::numeric_limits<double>::max()) {}
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
};
#endif // ! RAY_H

