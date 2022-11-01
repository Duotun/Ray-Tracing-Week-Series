#pragma once
#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "utility.hpp"
#include "hittable.hpp"
#include "material.hpp"
#include "texture.hpp"

class constant_medium : public hittable {
public:
	constant_medium(shared_ptr<hittable> b, double d, shared_ptr<texture> a)
		: boundary(b),
		neg_inv_density(-1/d),   //for multiplying with log, mass 1 for volume
		phase_function(make_shared<isotropic>(a)) {}
	constant_medium(shared_ptr<hittable> b, double d, color c)
		: boundary(b),
		neg_inv_density(-1 / d),
		phase_function(make_shared<isotropic>(c))
	{}

	virtual bool Intersect(ray& r, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
		return boundary->bounding_box(time0, time1, output_box);
	}
	//members
	shared_ptr<hittable> boundary;
	shared_ptr<material> phase_function;
	double neg_inv_density;
};

bool constant_medium::Intersect(ray& r, hit_record& rec) const {
	// Print occasional samples when debugging. To enable, set enableDebug true.
	const bool enableDebug = false;
	const bool debugging = enableDebug && random_double() < 0.00001;
	//ray tmpr = r;

	hit_record rec1, rec2;
	// make sure two records for the volume estimation
	auto tmpmin = r.m_tmin, tmpmax = r.m_tmax;
	r.m_tmin = -infinity;
	r.m_tmax = infinity;
	if (!boundary->Intersect(r, rec1))
		return false;
	r.m_tmin = rec1.t + 0.0001;
	//std::cerr << "Here: \n";
	r.m_tmax = infinity;
	if (!boundary->Intersect(r, rec2))  //assume convex 
		return false;
	
	if (debugging) std::cerr << "\nt_min=" << rec1.t << ", t_max=" << rec2.t << '\n';

	if (rec1.t < tmpmin) rec1.t = tmpmin;
	if (rec2.t > tmpmax) rec2.t = tmpmax;

	r.m_tmin = tmpmin;
	r.m_tmax = tmpmax;  //resume the ray segmentation ????
	
	
	if (rec1.t >= rec2.t) return false;
	if (rec1.t < 0)  rec1.t = 0;  //forward ray
	

	const auto ray_length = r.dir.length();  //it is fine we just use unit dir vector - 1 but keep the consistence
	const auto dist_inside_boundary = (rec2.t - rec1.t) * ray_length;   //assum convex shape
	const auto hit_distance = neg_inv_density * log(random_double());  // negative * negative = positive
	
	//std::cerr << "\n Dist: " << dist_inside_boundary << "\n";
	if (hit_distance > dist_inside_boundary)
		return false;

	rec.t = rec1.t + hit_distance / ray_length;
	rec.p = r.at(rec.t);

	//r.m_tmax = rec.t;

	if (debugging) {
		std::cerr << "hit_distance = " << hit_distance << '\n'
			<< "rec.t = " << rec.t << '\n'
			<< "rec.p = " << rec.p << '\n';
	}

	rec.normal = Vector3(1, 0, 0);  // arbitrary
	rec.front_face = true;     // also arbitrary
	rec.mat_ptr = phase_function;

	return true;
}

#endif // !CONSTANT_MEDIUM_H
