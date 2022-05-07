#pragma once
#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "hitable.hpp"
#include "utility.hpp"
#include <cstdint>
//#include "material.hpp"


class sphere : public hitable {
public:
	__device__ sphere() { radius = 1.0f; }
	__device__ sphere(Vector3 cen, float r, material *m): center(cen), radius(r), mat_ptr(m) {}
	__device__ virtual bool hit(ray& r, hit_record& rec) const;

	Vector3 center;
	float radius;
	material* mat_ptr;
};

__device__ bool sphere::hit(ray& r, hit_record& rec) const {
	const Vector3 op = center - r.org;
	const double dop = dot(unit_vector(r.dir), op);
	const double D = dop * dop - dot(op, op) + (double)radius * radius;

	if (0.0 > D) {   //D is determinant
		return false;
	}

	const double sqrtD = sqrt(D);
	double tmin = dop - sqrtD;

	//no intersection at all, two results are not in the range
	if (tmin <= r.m_tmin || tmin >= r.m_tmax)
	{
		tmin = dop + sqrtD;  //test the second result
		if (tmin <= r.m_tmin || tmin >= r.m_tmax)
			return false;

	}

	rec.t = (float)tmin;
	rec.p = r.at(rec.t);
	Vector3 outward_normal = unit_vector((rec.p - center) / radius);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;   //assign the material

	return true;

}

#endif