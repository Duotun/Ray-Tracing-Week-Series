#pragma once

#ifndef HITTABLELIST_H
#define HITTABLELIST_H
#include "hittable.hpp"
#include <memory>   //shared_ptr
#include <vector>


using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {

public:
	std::vector<shared_ptr<hittable>> objects;
public:
	hittable_list() {}
	hittable_list(shared_ptr<hittable>object) noexcept { add(object); }
	virtual bool bounding_box(
		double time0, double time1, aabb& output_box) const override;
	virtual double pdf_value(const point3& o, const Vector3& v) const override;
	virtual Vector3 random(const Vector3& o) const override;
	void clear() { objects.clear(); }
	void add(shared_ptr<hittable>object) { objects.push_back(object); }
	bool Intersect(ray& r, hit_record& rec) const
	{
		//used to check whether anything is hit
		hit_record temp_rec;
		bool hit_anything = false;
		for (const auto& object : objects)
		{
			if (object->Intersect(r, temp_rec))
			{
				hit_anything = true; //I need to update the t_max for the nearest ray
				r.m_tmax = temp_rec.t;  
				rec = temp_rec;
			}
		}

		return hit_anything;
	}
};

bool hittable_list::bounding_box(double time0, double time1, aabb& output_box) const
{
	//compute bounding box for the list on the fly
	if (objects.empty()) return false;
	bool first_box = true;   // first box doean't need to expand the bbox
	aabb temp_box;
	for (const auto &object: objects)
	{
		if (!object->bounding_box(time0, time1, temp_box)) return false;
		output_box = first_box ? temp_box : surrounding_box(temp_box, output_box);
		first_box = false;
	}
	return true;   
}

double hittable_list::pdf_value(const point3& o, const Vector3& v) const
{
	auto weight = 1.0 / objects.size();
	auto sum = 0.0;

	for (const auto& object : objects)
		sum += weight * object->pdf_value(o, v);  //put mixed pdf here

	return sum;
}

Vector3 hittable_list::random(const Vector3& o) const {
	auto int_size = static_cast<int>(objects.size());
	return objects[random_int(0, int_size - 1)]->random(o);
}
#endif // !HITTABLELIST_H
