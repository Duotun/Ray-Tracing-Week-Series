
#pragma region
//includes
#include <iostream>
#include "utility.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hittablelist.hpp"
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
color ray_color_world(ray& r, const hittable& world)
{
	hit_record h;
	if (world.Intersect(r, h))   //draw the sphere normals, sphere is at (0, 0, -1)
	{
		return 0.5 * color(h.normal + color(1, 1, 1));
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

	//World
	hittable_list world;
	world.add(make_shared<Sphere>(0.5, point3(0, 0, -1)));
	world.add(make_shared<Sphere>(100, point3(0, -100.5, -1)));
	//scene description
	Sphere sphere(0.5, point3(0.0, 0.0, -1.0), point3(0, 0, 0), point3(0, 0, 0));


	//Camera setting in the camera coordinates afer projection as well?
	double viewport_height = 2.0;
	double viewport_width = aspect_ratio * viewport_height;
	double focal_length = 1.0;

	auto origin = point3(0, 0, 0);
	auto horizontal = Vector3(viewport_width, 0, 0);
	auto vertical = Vector3(0, viewport_height, 0);   //2.0
	Vector3 lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vector3(0, 0, focal_length);
	
	
	//render into the image .ppm format
	std::cout << "P3\n" << width << ' ' << height << "\n255\n";
	//wirte the image from left to right, top to bottom (first row to the last row)
	//so pmm still defines bottom-left at (0, 0)
	for (int i = height - 1; i >= 0; --i)  // bottom becomes black
	{
		std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
		for (int j = 0; j < width; ++j)
		{
			auto u = double(j) / (width-1.0);
			auto v = double(i) / (height-1.0);
			//double b = 0.25;
			ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin); // shoot to the screen
			//color pixel_color(ray_color_sphere(r,sphere));
			//color pixel_color(ray_color_sphere_mapnormals(r, sphere));
			color pixel_color(ray_color_world(r,world));
			//color pixel_color(ray_color(r));
	
			//simple ppm format, tone map to 0-255
			write_color(std::cout, pixel_color);
		}
	}
	std::cerr << "\nDone.\n";
	return 0;
}
