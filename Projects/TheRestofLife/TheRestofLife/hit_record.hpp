#pragma once
#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include "utility.hpp"
#include <memory>    //for shared_ptr
#include "material.hpp"

class material;   //alert the compiler that the pointer is to a class, which the ¡°class material¡±

class hit_record {
public:
	point3 p;
	Vector3 normal;
	shared_ptr<material> mat_ptr;   //hit record needs to know materials each other
	double t;
	double u;
	double v;   // add uv coordinates of the ray object hit point
	bool front_face;   // for normal sides whether pointing out

	inline void set_face_normal(const ray& r, const Vector3& outward_normal)
	{
		front_face = dot(unit_vector(r.direction()), outward_normal) < 0.0;   // <0 outside the sphere, make normals pointing out always
		normal = front_face ? outward_normal : -outward_normal;
	}

};
#endif // !HIT_RECORD_H