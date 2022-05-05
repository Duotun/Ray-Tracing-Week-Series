#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"

class camera {

public:
	__device__ camera()
	{
		lower_left_corner = Vector3(-2.0f, -1.0f, -1.0f);
		horizontal = Vector3(4.0f, 0.0f, 0.0f);
		vertical = Vector3(0.0f, 2.0f, 0.0f);
		origin = Vector3(0.0f, 0.0f, 0.0f);
	}

	__device__ ray get_ray(float u, float v)
	{
		return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	}

	Vector3 origin;
	Vector3 lower_left_corner;
	Vector3 horizontal;
	Vector3 vertical;

};

#endif // ! CAMERA_H
