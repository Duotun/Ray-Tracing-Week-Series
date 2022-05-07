#pragma once
#ifndef HIT_RECORD_H
#define HIT_RECORD_H

//class material;

#include "utility.hpp"
#include "vector.hpp"
#include "ray.hpp"


class material;
class hit_record {
public:
	float t;
	Vector3 p;
	Vector3 normal;
	material* mat_ptr;
	bool front_face;   // for normal sides whether pointing out
	__device__ hit_record() { t = 0.0; p = normal = Vector3{ 0.0f, 0.0f, 0.0f }; front_face = false; mat_ptr = nullptr; }
	__device__ ~hit_record() = default;

	__device__ void set_face_normal(const ray& r, const Vector3& outward_normal)
	{
		front_face = dot(r.dir, outward_normal) < 0.0;   // <0 outside the sphere, make normals pointing out always
		normal = front_face ? outward_normal : -outward_normal;
	}
};

#include "material.hpp"   // make the definition and implementation a little bite in sequence

#endif // !HIT_RECORD_H