#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H
#include "ray.hpp"

struct hit_record {
	point3 p;
	Vector3 normal;
	double t;
	bool front_face;   // for normal sides whether pointing out

	inline void set_face_normal(const ray& r, const Vector3& outward_normal)
	{
		front_face = dot(unit_vector(r.direction()), outward_normal) < 0.0;   // <0 outside the sphere, make normals pointing out always
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
	
	public:
		virtual bool Intersect(ray& r, hit_record& rec) const = 0;
};

#endif // !HITTABLE_H

