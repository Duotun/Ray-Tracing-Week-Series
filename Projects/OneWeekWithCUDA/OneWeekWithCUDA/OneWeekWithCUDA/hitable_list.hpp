#pragma once
#ifndef HITABLE_LIST_H
#define HITABLE_LIST_H

#include "hitable.hpp"

class hitable_list : public hitable {
public:
	__device__ hitable_list() { list == NULL; list_size = 0; }
	__device__ hitable_list(hitable** l, int n) { list = l; list_size = n; }
	__device__ virtual bool hit(ray& r, hit_record& rec) const;
	hitable ** list;
	int list_size;


};


__device__ bool hitable_list::hit(ray& r, hit_record& rec) const {

	hit_record tmp_rec;
	bool hit_anything = false;
	for (auto i = 0; i < list_size; i++)
	{
		if (list[i]->hit(r, tmp_rec))
		{
			hit_anything = true;
			rec = tmp_rec;  //if hit update the hit record
		}
	}

	return hit_anything;
}

#endif // ! HITABLE_LIST_H

