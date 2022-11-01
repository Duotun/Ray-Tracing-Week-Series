#pragma once

#ifndef HITTABLE_H
#define HITTABLE_H
#include "ray.hpp"
#include "utility.hpp"
#include "material.hpp"
#include <memory> 
#include "hit_record.hpp"
#include "aabb.hpp"

class hittable {

public:
	virtual bool Intersect(ray& r, hit_record& rec) const = 0;
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const = 0;  //time interval for moving objects
};

class translate : public hittable {
public:
	translate(shared_ptr<hittable> p, const Vector3 & displacement)
		: ptr(p), offset(displacement){}

	virtual bool Intersect(ray& r, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

	//members;
	shared_ptr<hittable> ptr;
	Vector3 offset;
};

bool translate::Intersect(ray& r, hit_record& rec) const {
	// translate the ray in the recerted direction
	ray moved_r(r.org - offset, r.dir, r.time());
	moved_r.m_tmin = r.m_tmin;   //to fit with the starting t change potentially
	moved_r.m_tmax = r.m_tmax;
	if (!ptr->Intersect(moved_r, rec))
		return false;

	//back the hitting position to indicate the object's move
	rec.p += offset;
	rec.set_face_normal(moved_r, rec.normal);
	return true;
}

bool translate::bounding_box(double time0, double time1, aabb& output_box) const {
	if (!ptr->bounding_box(time0, time1, output_box))
		return false;
	output_box = aabb(output_box.min() + offset, output_box.max() + offset);
	return true;
}

//rotate the ray and apply to the objects back
//for the bounding box, we need to extend the box
class rotate_y : public hittable {

public:
	rotate_y(shared_ptr<hittable> p, double angle);
	virtual bool Intersect(ray& r, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override
	{
		output_box = bbox;
		return hasbox;
	}
	//members;
	shared_ptr<hittable> ptr;
	double sin_theta;
	double cos_theta;
	bool hasbox;
	aabb bbox;
};

rotate_y::rotate_y(shared_ptr<hittable> p, double angle)
{
	auto radians = degrees_to_radians(angle);
	sin_theta = sin(radians);
	cos_theta = cos(radians);

	ptr = p;  //assign the hittable object
	hasbox = ptr->bounding_box(0, 1, bbox);
	point3 min(infinity, infinity, infinity);
	point3 max(-infinity, -infinity, -infinity);

	//fill all three dimensions, 3D Case
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				auto x = i * bbox.max().x() + (1 - i) * bbox.min().x();  //test to choose which one
				auto y = j * bbox.max().y() + (1 - j) * bbox.min().y();
				auto z = k * bbox.max().z() + (1 - k) * bbox.min().z();

				//perform the rotation for the box
				auto newx = cos_theta * x + sin_theta * z;
				auto newz = -sin_theta * x + cos_theta * z;

				Vector3 tester(newx, y, newz);
				for (int c = 0; c < 3; c++)
				{
					min[c] = fmin(min[c], tester[c]);
					max[c] = fmax(max[c], tester[c]);
				}
			}
		}
	}

	bbox = aabb(min, max);
}

bool rotate_y::Intersect(ray& r, hit_record& rec) const {
	//return false;
	auto origin = r.origin();
	auto direction = r.dir;

	//perform the ray rotation
	origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
	origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

	direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
	direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

	ray rotated_r(origin, direction, r.time());
	rotated_r.m_tmin = r.m_tmin;
	rotated_r.m_tmax = r.m_tmax;
	if (!ptr->Intersect(rotated_r, rec))
		return false;
	
	//rotate the collision record back
	auto p = rec.p;
	auto normal = rec.normal;

	p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
	p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

	normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
	normal[2] = -sin_theta* rec.normal[0] + cos_theta * rec.normal[2];

	rec.p = p;
	rec.set_face_normal(rotated_r, normal);  //define the rotaiton with rotated_r

	return true;
}
#endif // !HITTABLE_H
