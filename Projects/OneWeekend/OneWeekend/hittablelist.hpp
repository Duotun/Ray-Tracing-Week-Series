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
	hittable_list(){}
	hittable_list(shared_ptr<hittable>object) noexcept { add(object); }

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
				hit_anything = true;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}
};

#endif // !HITTABLELIST_H
