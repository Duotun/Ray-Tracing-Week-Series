#pragma once
#ifndef CAMERA_H
#define CAMERA_H
#include "utility.hpp"

class camera {
public:
	//vertical fov in degress, decide the viewport
	//look from, look at to define the viewing direction, and vup for camera rotation + cross product
	camera(
		point3 lookfrom,
		point3 lookat,
		Vector3 vup,
		double vfov,
		double aspect_ratio,
		double aperture,
		double focus_dist,
		double _time0,    //sending rays over the time period [_time0, _time1]
		double _time1) {

		auto theta = degrees_to_radians(vfov);
		auto h = tan(theta / 2);
		double viewport_height = 2.0 * h;
		double viewport_width = aspect_ratio * viewport_height;

		w = unit_vector(lookfrom - lookat);    // positive z direction remember is backwards from the plane
		u = unit_vector(cross(vup, w));
		v = unit_vector(cross(w, u));


		origin = lookfrom;    //lookfrom is the camera's coordinates
		horizontal = focus_dist * viewport_width * u;   //because we are also focus_dist away, the ray range should be also enlarged
		vertical = focus_dist * viewport_height * v;  // 
		lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;  //shoot toward the focus plane at the length of focus_dist 
		//however, only at the focus_distance, everything will be clear
		lens_radius = aperture / 2.0;

		time0 = _time0;
		time1 = _time1;

	}
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

	ray get_ray(double s, double t) const {
		Vector3 rd = lens_radius * random_in_unit_disk();
		//std::cerr << "lens_radius: " << lens_radius << std::endl;
		Vector3 offset = rd.x() * u + rd.y() * v;
		//std::cerr << "offset: " << offset << std::endl;
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset
		, random_double(time0, time1));   //the ray from camera varies with time as well to produce motion blur
	}

	point3 origin;
	point3 lower_left_corner;
	Vector3 horizontal;
	Vector3 vertical;
	Vector3 u, v, w;
	double lens_radius = 1.0;
	double time0, time1;  //shutter open/close time

};


#endif // !CAMERA_H

