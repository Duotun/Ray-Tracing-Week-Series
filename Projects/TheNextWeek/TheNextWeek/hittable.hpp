#pragma once

#ifndef HITTABLE_H
#define HITTABLE_H
#include "ray.hpp"
#include "utility.hpp"
#include "material.hpp"
#include <memory> 
#include "hit_record.hpp"

class hittable {

public:
	virtual bool Intersect(ray& r, hit_record& rec) const = 0;
};

#endif // !HITTABLE_H

