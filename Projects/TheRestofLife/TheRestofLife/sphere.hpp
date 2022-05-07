#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "utility.hpp"
#include "onb.hpp"
#include "hittable.hpp"

inline Vector3 random_to_sphere(double radius, double dis_squared)
{
	auto r1 = random_double();  //from random numbers to vector
	auto r2 = random_double();
	auto z = 1 + r2 * (sqrt(1 - radius * radius / dis_squared) - 1.0);

	auto phi = 2 * pi * r1;
	auto x = cos(phi) * sqrt(1 - z * z);
	auto y = sin(phi) * sqrt(1 - z * z);
	return Vector3(x, y, z);
}

class sphere : public hittable {
public:

	sphere();
	sphere(point3 cen, double r, shared_ptr<material> m) :
		center(cen), radius(r), mat_ptr(m) {};

	virtual bool Intersect(ray& r, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;  //time interval for moving objects
	virtual double pdf_value(const point3& o, const Vector3& v) const override;
	virtual Vector3 random(const Vector3& o) const override;
	//members
	point3 center;
	double radius;
	shared_ptr<material> mat_ptr;

private:
	static void get_sphere_uv(const point3& p, double& u, double& v) {
		// p: a given point on the sphere of radius one, centered at the origin.
		// u: returned value [0,1] of angle around the Y axis from X=-1.
		// v: returned value [0,1] of angle from Y=-1 to Y=+1.
		//     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
		//     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
		//     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

		auto theta = acos(-p.y());
		auto phi = atan2(-p.z(), p.x()) + pi;

		u = phi / (2 * pi);
		v = theta / pi;
	}


};

double sphere::pdf_value(const point3& o, const Vector3& v) const
{
	hit_record rec;
	ray test_ray(o, v);
	if (!this->Intersect(test_ray, rec))
		return 0;
	auto cos_theta_max = sqrt(1.0 - radius * radius / (center - o).length_squared());
	auto solid_angle = 2 * pi * (1 - cos_theta_max);

	return 1 / solid_angle;
}

Vector3 sphere::random(const point3& o) const {
	Vector3 direction = center - o;
	auto distance_squared = direction.length_squared();
	onb uvw;
	uvw.build_from_w(direction);
	return uvw.local(random_to_sphere(radius, distance_squared));

}

bool sphere::Intersect(ray& r, hit_record& rec) const
{
	const Vector3 op = center - r.org;
	const double dop = dot(unit_vector(r.dir), op);
	const double D = dop * dop - dot(op, op) + radius* radius;

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

	//r.m_tmax = tmin;  //update m_tmax in the hittable_list
	rec.t = tmin;
	rec.p = r.at(rec.t);
	Vector3 outward_normal = unit_vector((rec.p - center) / radius);
	//rec.normal = outward_normal;
	rec.set_face_normal(r, outward_normal);   //normalized normal
	get_sphere_uv(outward_normal, rec.u, rec.v);
	rec.mat_ptr = mat_ptr;  // use pointer to pass values
	return true;
}

bool sphere::bounding_box(double time0, double time1, aabb& output_box) const
{
	output_box = aabb(center - Vector3(radius, radius, radius),
		center + Vector3(radius, radius, radius));
	return true;
}

#endif
