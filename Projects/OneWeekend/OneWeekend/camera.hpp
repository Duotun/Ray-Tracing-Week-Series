#pragma once
#ifndef CAMERA_H
#define CAMERA_H
#include "utility.hpp"

class camera {
public:
	camera() {
		auto aspect_ratio = 16.0 / 9.0;
		double viewport_height = 2.0;
		double viewport_width = aspect_ratio * viewport_height;
		double focal_length = 1.0;

		origin = point3(0, 0, 0);
		horizontal = Vector3(viewport_width, 0, 0);
		vertical = Vector3(0, viewport_height, 0);   //2.0
		lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vector3(0, 0, focal_length);
	}

	camera(point3 org) {

		auto aspect_ratio = 16.0 / 9.0;
		double viewport_height = 2.0;
		double viewport_width = aspect_ratio * viewport_height;
		double focal_length = 1.0;

		origin = org;
		horizontal = Vector3(viewport_width, 0, 0);
		vertical = Vector3(0, viewport_height, 0);   //2.0
		lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vector3(0, 0, focal_length);

	}

	~camera() = default;

	ray get_ray(double u, double v) const {
		return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	}

	point3 origin;
	point3 lower_left_corner;
	Vector3 horizontal;
	Vector3 vertical;

};


#endif // !CAMERA_H

