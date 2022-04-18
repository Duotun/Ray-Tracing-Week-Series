
#pragma region
//includes
#include <iostream>
#include "utility.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hittablelist.hpp"
#include "camera.hpp"
#pragma endregion

//indicate the image resolution
 int width = 256;
 int height = 256;
 double aspect_ratio = 16.0 / 9.0;

//as a gradient background depending on the height of the y coordinate
color ray_color(const ray& r)
{
	Vector3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

//to color a sphere for identifying intersections
color ray_color_sphere(ray& r, const Sphere& sphere)
{
	hit_record h;
	if (sphere.Intersect(r,h))   //draw the sphere
		return color(1.0, 0.0, 0.0);
	Vector3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

//to color a sphere for identifying intersections & simple normal visualization
color ray_color_sphere_mapnormals(ray& r, const Sphere& sphere)
{	
	hit_record h;
	if (sphere.Intersect(r,h))   //draw the sphere normals, sphere is at (0, 0, -1)
	{
		Vector3 N = h.normal;
		//std::cerr << "Sphere Oriign: " << sphere.m_p << std::endl;
		return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
	}
	Vector3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

//world drawing
color ray_color_world(ray& r, const hittable& world, int depth)
{
	if (depth <= 0)
	{
		return color(0, 0, 0);
	}

	hit_record h;
	if (world.Intersect(r, h))   //draw the sphere normals, sphere is at (0, 0, -1)
	{
		point3 target = h.p + h.normal + random_in_unit_sphere();
		ray next(h.p, Vector3(target - h.p));
		return  0.5 * ray_color_world(next, world, depth-1);
	}
	Vector3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
	//image setting
	aspect_ratio = 16.0 / 9.0;
	width = 400;
	height = static_cast<int> (width / aspect_ratio);
	const int samples_per_pixel = 100;
	const int max_depth = 50;

	//World
	hittable_list world;
	world.add(make_shared<Sphere>(0.5, point3(0, 0, -1)));
	world.add(make_shared<Sphere>(100, point3(0, -100.5, -1)));
	//scene description
	Sphere sphere(0.5, point3(0.0, 0.0, -1.0), point3(0, 0, 0), point3(0, 0, 0));


	//Camera camera
	camera cam;
	
	
	
	//render into the image .ppm format
	std::cout << "P3\n" << width << ' ' << height << "\n255\n";
	//wirte the image from left to right, top to bottom (first row to the last row)
	//so pmm still defines bottom-left at (0, 0)
	for (int i = height - 1; i >= 0; --i)  // bottom becomes black
	{
		std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
		for (int j = 0; j < width; ++j)
		{
			
			//double b = 0.25;
			color pixel_color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; s++) {
				auto u = double(j + random_double()) / (width - 1.0);
				auto v = double(i + random_double()) / (height - 1.0);
				ray r = cam.get_ray(u, v);
				pixel_color += ray_color_world(r, world, max_depth);
			}
			
			//ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin); // shoot to the screen
			//color pixel_color(ray_color_sphere(r,sphere));
			//color pixel_color(ray_color_sphere_mapnormals(r, sphere));
			//color pixel_color(ray_color_world(r,world));
			//color pixel_color(ray_color(r));
	
			//simple ppm format, tone map to 0-255 + anti-aliasing
			write_color(std::cout, pixel_color,samples_per_pixel);
		}
	}
	std::cerr << "\nDone.\n";
	return 0;
}
