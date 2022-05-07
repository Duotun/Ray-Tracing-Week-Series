#pragma once
#ifndef  HITABLE_H
#define HITABLE_H
#include "ray.hpp"
#include "hit_record.hpp"
#include <memory>


class hitable {

public:
	__device__ virtual bool hit(ray& r, hit_record& rec) const = 0;
};


#endif // ! HITTABLE_H


