#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"


__device__ Vector3 random_in_unit_disk(curandState* local_rand_state)
{
	Vector3 p;
	do {
		p = 2.0f * Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f) - Vector3(1.0f, 1.0f, 0.0f);
	} while (dot(p, p) >= 1.0f);

    return p;
}

class camera {

public:

	__device__ camera(Vector3 lookfrom, Vector3 lookat, Vector3 up, float vfov, float aspect,
		float aperture, float focus_dist)
	{
		lens_radius = aperture / 2.0f;

		float theta = degrees_to_radians(vfov);
		float  h = tanf(theta / 2.0f);
		float viewport_height = 2.0 * h;
		float viewport_width = aspect * viewport_height;

		 w = unit_vector(lookfrom - lookat);   //-z
		 u = unit_vector(cross(up, w));
		 v = cross(w, u);

		 origin = lookfrom;
		 horizontal = focus_dist * viewport_width * u;  //focus plane is simulate by focus_distance
		 vertical = focus_dist * viewport_height * v;
		 lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist*w;
	}
	__device__ camera()
	{
		lower_left_corner = Vector3(-2.0f, -1.0f, -1.0f);
		horizontal = Vector3(4.0f, 0.0f, 0.0f);
		vertical = Vector3(0.0f, 2.0f, 0.0f);
		origin = Vector3(0.0f, 0.0f, 0.0f);
	}

	__device__ ray get_ray(float s, float t, curandState* local_rand_state)
	{
		Vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		Vector3 offset = rd.x()* u + rd.y() * v;
		return ray(origin+offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

	Vector3 origin;
	Vector3 lower_left_corner;
	Vector3 horizontal;
	Vector3 vertical;
	Vector3 u, v, w;
	double lens_radius = 1.0;

};

#endif // ! CAMERA_H
