#pragma once
#ifndef BVH_H
#define BVH_H
#include "utility.hpp"
#include "hittable.hpp"
#include "hittablelist.hpp"

inline bool box_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis)
{
	aabb box_a;
	aabb box_b;

	if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
		std::cerr << "No bounding box in bvh_node constructor.\n";

	return box_a.min().p[axis] < box_b.min().p[axis];
}

inline bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b)
{
	return box_compare(a, b, 0);
}

inline bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b)
{
	return box_compare(a, b, 1);
}

inline bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b)
{
	return box_compare(a, b, 2);
}

class bvh_node : public hittable {
public:
	bvh_node() noexcept = default;
	bvh_node(const hittable_list& list, double time0, double time1)
		: bvh_node(list.objects, 0, list.objects.size(), time0, time1){}
	
	//the most complicated part, build the bvh tree
	//start end for the objects' list length
	bvh_node(const std::vector<shared_ptr<hittable>>& src_objects,
		size_t start, size_t end, double time0, double time1) {

		//divide into two sub-list along one axis
		//sort and put half in each subtree with random axis

		auto objects = src_objects;
		int axis = random_int(0, 2);  //x, y, or z

		auto comparator = (axis == 0) ? box_x_compare :
			(axis == 1) ? box_y_compare : box_z_compare;

		size_t object_span = end - start;
		if (object_span == 1)   //only one object, duplicate
		{
			left = right = objects[start];
		}
		else if (object_span == 2)  //left one, right ont
		{
			if (comparator(objects[start], objects[start+1]))
			{
				left = objects[start];
				right = objects[start+1];
			}
			else {
				right = objects[start];
				left = objects[start+1];
			}
		}
		else
		{
			//perform the sort
			std::sort(objects.begin() + start, objects.begin() + end, comparator);
			auto mid = start + object_span / 2;
			left = make_shared<bvh_node>(objects, start, mid, time0, time1);
			right = make_shared<bvh_node>(objects, mid+1, end, time0, time1);
		}

		//assign the final box of the node
		aabb bleft, bright;
		if (!left->bounding_box(time0, time1, bleft) ||
			!right->bounding_box(time0, time1, bright))
			std::cerr << "No bounding box in bvh_node constructor.\n";

		box = surrounding_box(bleft, bright);  //build the box from left and right children

	}

	virtual bool Intersect(ray& r, hit_record& rec) const override;

	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
	//members
	shared_ptr<hittable> left;
	shared_ptr<hittable> right; //use hittable represent more general cases
	aabb box;  //the node's box
};

bool bvh_node::Intersect(ray& r, hit_record& rec) const
{
	//check the node, if hit, go to the children
	if (!box.hit(r)) return false;   //check the current bounding box

	bool hit_left = left->Intersect(r, rec);
	if(hit_left) r.m_tmax = rec.t;
	bool hit_right = right->Intersect(r, rec);

	return hit_left || hit_right;
}

bool bvh_node::bounding_box(double time0, double time1, aabb& output_box) const
{
	output_box = box;
	return true;
}

#endif // ! BVH_H
