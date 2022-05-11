#pragma once
#ifndef  AARECT_H
#define AARECT_H

#include "utility.hpp"
#include "hittable.hpp"

//simulate the rectangule with a little
class xy_rect : public hittable {
public:
	xy_rect() {}
	xy_rect(double _x0, double _x1, double _y0, double _y1, double _k,
		shared_ptr<material> mat)
		:x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

	virtual bool Intersect(ray& r, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

	shared_ptr<material> mp;
	double x0, x1, y0, y1, k;  // k as the thin z
};
bool xy_rect::Intersect(ray& r, hit_record& rec) const {
	auto t = (k - r.origin().z()) / r.direction().z();
	//if(r.m_tmin>0.0001)
	//	std::cerr << "\n tmin: " << r.m_tmin << "\n";
	if (t < r.m_tmin || t > r.m_tmax)
		return false;

	auto x = r.origin().x() + t * r.direction().x();
	auto y = r.origin().y() + t * r.direction().y();
	if (x < x0 || x > x1 || y < y0 || y > y1)
		return false;

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	auto outward_normal = Vector3(0, 0, 1);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);

	return true;
}

bool xy_rect::bounding_box(double time0, double time1, aabb& output_box) const
{
	//the bounding box must have non-zero width, so pad the z dimension a little bit
	//dimension a small amount
	output_box = aabb(point3(x0, y0, k - 0.0001), point3(x1, y1, k + 0.0001));
	return true;
}

class xz_rect : public hittable {
public:
	xz_rect() {}
	xz_rect(double _x0, double _x1, double _z0, double _z1, double _k,
		shared_ptr<material> mat)
		: x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};  //k is for y now

	virtual bool Intersect(ray& r, hit_record& rec) const override;

	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
		// The bounding box must have non-zero width in each dimension, so pad the Y
		// dimension a small amount.
		output_box = aabb(point3(x0, k - 0.0001, z0), point3(x1, k + 0.0001, z1));
		return true;
	}

	// v is from a random value
	virtual double pdf_value(const point3& origin, const Vector3& v) const override {
		hit_record rec;
		ray test_ray(origin, v);
		if (!this->Intersect(test_ray, rec))   //for lighting up directly
			return 0.0;  // simulate shadow ray test 

		auto area = (x1 - x0) * (z1 - z0);
		auto distance_squared = rec.t * rec.t * v.length_squared();
		auto cosine = fabs(dot(v, rec.normal)) / v.length();
		return distance_squared / (cosine * area);

	}

	virtual Vector3 random(const point3& origin) const override {
		auto random_point = point3(random_double(x0, x1), k, random_double(z0, z1));
		return random_point - origin;   // shooting out
	}

public:
	shared_ptr<material> mp;
	double x0, x1, z0, z1, k;

};

bool xz_rect::Intersect(ray& r, hit_record& rec) const {

	auto t = (k - r.origin().y()) / r.direction().y();
	if (t < r.m_tmin|| t>r.m_tmax)
		return false;
	auto x = r.origin().x() + t * r.direction().x();
	auto z = r.origin().z() + t * r.direction().z();
	if (x < x0 || x > x1 || z < z0 || z > z1)
		return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	auto outward_normal = Vector3(0, 1, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;
}

class yz_rect : public hittable {

public:
	yz_rect() {}

	yz_rect(double _y0, double _y1, double _z0, double _z1, double _k,
		shared_ptr<material> mat)
		: y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

	virtual bool Intersect(ray& r, hit_record& rec) const override;

	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
		// The bounding box must have non-zero width in each dimension, so pad the X
		// dimension a small amount.
		output_box = aabb(point3(k - 0.0001, y0, z0), point3(k + 0.0001, y1, z1));
		return true;
	}

public:
	shared_ptr<material> mp;
	double y0, y1, z0, z1, k;
};

bool yz_rect::Intersect(ray& r, hit_record& rec) const {

	auto t = (k - r.origin().x()) / r.direction().x();
	if (t < r.m_tmin || t > r.m_tmax)
		return false;
	auto y = r.origin().y() + t * r.direction().y();
	auto z = r.origin().z() + t * r.direction().z();
	if (y < y0 || y > y1 || z < z0 || z > z1)
		return false;
	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	auto outward_normal = Vector3(1, 0, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;
}
#endif // ! AARECT_H
