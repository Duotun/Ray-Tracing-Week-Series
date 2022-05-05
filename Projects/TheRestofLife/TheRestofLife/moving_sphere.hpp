#pragma once
#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "utility.hpp"
#include "hittable.hpp"

class moving_sphere : public hittable {
public:
	moving_sphere():time0(0.0), time1(1.0), radius(1.0) {};
	moving_sphere(point3 cen0, point3 cen1,
		double _time0, double _time1,
		double r,
		shared_ptr<material> m):
		center0(cen0), center1(cen1),
		time0(_time0), time1(_time1),
		radius(r), mat_ptr(m){};

	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
	~moving_sphere() = default;
	virtual bool Intersect(ray& r, hit_record& rec) const override;
	point3 center(double time) const;

public:
	point3 center0, center1;
	double time0, time1;    // for the center moving
	double radius;
	shared_ptr<material>mat_ptr;
};

point3 moving_sphere::center(double time) const {

	return  center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

[[nodiscard]]
bool moving_sphere::Intersect(ray& r, hit_record& rec) const {
	const Vector3 op = center(r.time()) - r.org;
	const double dop = dot(unit_vector(r.dir), op);
	const double D = dop * dop - dot(op, op) + radius * radius;

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

	r.m_tmax = tmin;
	rec.t = r.m_tmax;
	rec.p = r.at(rec.t);
	Vector3 outward_normal = unit_vector((rec.p - center(r.time())) /radius);
	//rec.normal = outward_normal;
	rec.set_face_normal(r, outward_normal);   //normalized normal
	rec.mat_ptr = mat_ptr;  // use pointer to pass values
	return true;
}

[[nodiscard]]
bool moving_sphere::bounding_box(double time0, double time1, aabb& output_box) const
{
	// use two times to construct the two limits
	aabb box0(
		center(time0) - Vector3(radius, radius, radius),
		center(time0) + Vector3(radius, radius, radius)
	);

	aabb box1(
		center(time1) - Vector3(radius, radius, radius),
		center(time1) + Vector3(radius, radius, radius)
	);

	output_box = surrounding_box(box0, box1);
	return true;
}
#endif 