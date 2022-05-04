#pragma once
#ifndef AABB_H
#define AABB_H

#include "utility.hpp"

class aabb {
public:
	aabb() {}
	aabb(const point3& a, const point3& b) { minimum = a; maximum = b; }
	~aabb() = default;
	point3 min() const { return minimum; }
	point3 max() const { return maximum; }
	//I put the tmin and tmax in the ray constructor, start with t_min = 0, t_max = infinity
	bool hit(const ray& r) const {    
		double t_min = r.m_tmin;
		double t_max = r.m_tmax;

		// traverse the three dimensions
		for (int i = 0; i < 3; i++)
		{
			auto invD = 1.0 / r.dir[i];
			double t0 = (min()[i] - r.origin()[i]) * invD;
			double t1 = (max()[i] - r.origin()[i]) * invD;

			//if dir is not forward, reverse the result
			if (invD < 0.0)
				std::swap(t0, t1);
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max? t1: t_max;
			if (t_max <= t_min)
				return false;
		}
		return true;
	}
	point3 minimum;
	point3 maximum;
};

aabb surrounding_box(aabb& box0, aabb& box1)
{
	point3 small(std::fmin(box0.min().x(), box1.min().x()),
		std::fmin(box0.min().y(), box1.min().y()),
		std::fmin(box0.min().z(), box1.min().z()));

	point3 large(std::fmax(box0.max().x(), box1.max().x()),
		std::fmax(box0.max().y(), box1.max().y()),
		std::fmax(box0.max().z(), box1.max().z()));

	return aabb(small, large);

}
#endif