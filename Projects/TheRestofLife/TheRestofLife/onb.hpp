#pragma once
#ifndef ONB_H
#define ONB_H
#include "vector.hpp"
#include "utility.hpp"
class onb
{
public:
	onb() {}
	~onb() = default;

	inline Vector3 operator[](int i)const { return axis[i]; }
	Vector3 u() const { return axis[0]; }
	Vector3 v() const { return axis[1]; }
	Vector3 w() const { return axis[2]; }

	//local values based on this axes coordinate
	Vector3 local(double a, double b, double c)const {
		return a * u() + b * v() + c * w();
	}

	Vector3 local(const Vector3& a) const  
	{
		return a.x() * u() + a.y() * v() + a.z() * w();
	}
	void build_from_w(const Vector3& n);    //build orthonormal axes from provided one axis only
	Vector3 axis[3];   // three orthonormal axes
};

void onb::build_from_w(const Vector3& n)
{
	axis[2] = unit_vector(n);
	//just provide a random psudo vector for further cross products
	Vector3 a = (fabs(w().x()) > 0.9) ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
	axis[1] = unit_vector(cross(a, n));
	axis[0] = unit_vector(cross(n, axis[1]));
}
#endif // ! NVB_H
