#pragma once
#ifndef BOX_H
#define BOX_H

#include "utility.hpp"
#include "aarect.hpp"
#include "hittablelist.hpp"

class box : public hittable {
public: 
	  box() {}
	  box(const point3& p0, const point3& p1, shared_ptr<material> ptr);

	  virtual bool Intersect(ray& r, hit_record& rec) const override;
	  virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
		  output_box = aabb(box_min, box_max);
		  return true;
	  }
	  // members
	  point3 box_min;
	  point3 box_max;
	  hittable_list sides;  // six sides
};

box::box(const point3& p0, const point3& p1, shared_ptr<material> ptr) {
	box_min = p0;
	box_max = p1;

	//form 6 sides
	sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));
	sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));

	sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));
	sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));

	sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
	sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
}

bool box::Intersect(ray& r, hit_record& rec) const {
	return sides.Intersect(r, rec);
}
#endif // ! BOX_H

