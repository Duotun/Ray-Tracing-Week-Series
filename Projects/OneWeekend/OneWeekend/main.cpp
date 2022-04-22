
#pragma region
//includes
#include <iostream>
#include "utility.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hittablelist.hpp"
#include "camera.hpp"
#include "material.hpp"
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
	
	if (r.m_depth <= 0)
	{
		return color(0, 0, 0);
	}

	hit_record h;
	if (world.Intersect(r, h))   //draw the sphere normals, sphere is at (0, 0, -1)
	{
		ray scattered; color attenuation;
		if (h.mat_ptr->scatter(r, h, attenuation, scattered))
		{	
			scattered.m_depth = r.m_depth - 1;
			return attenuation * ray_color_world(scattered, world);
		}
		return color(0, 0, 0);  // this is for the metal materials, if no ourside rays, return black colors
		
	}
	Vector3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}


hittable_list random_scene() {
	hittable_list world;

	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<Sphere>(1000, point3(0, -1000, 0), ground_material));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = make_shared<lambertian>(albedo);
					world.add(make_shared<Sphere>(0.2, center, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<Sphere>(0.2, center, sphere_material));
				}
				else {
					// glass
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<Sphere>(0.2, center, sphere_material));
				}
			}
		}
	}

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<Sphere>(1.0, point3(0, 1, 0), material1));

	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<Sphere>( 1.0, point3(-4, 1, 0), material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<Sphere>(1.0, point3(4, 1, 0), material3));

	return world;
}




int main()
{
	// Image

	const auto aspect_ratio = 3.0 / 2.0;
	const int image_width = 1200;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int samples_per_pixel = 500;
	const int max_depth = 50;

	// World

	auto world = random_scene();

	// Camera

	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	Vector3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	auto aperture = 0.1;

	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

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
				ray r = cam.get_ray(u, v);  r.m_depth = max_depth;
				pixel_color += ray_color_world(r, world);
			}

			
			//simple ppm format, tone map to 0-255 + anti-aliasing
			write_color(std::cout, pixel_color, samples_per_pixel);
		}
	}
	std::cerr << "\nDone.\n";
	return 0;
}



/*
int main()   // test stage before the 13.sec
{
	//image setting
	aspect_ratio = 3.0 / 2.0;
	width = 400;
	height = static_cast<int> (width / aspect_ratio);
	const int samples_per_pixel = 100;
	const int max_depth = 50;

	//World
	hittable_list world;
	
	//scene description
	//define materials first
	auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
	//auto material_center = make_shared<lambertian>(color(0.7, 0.3, 0.3));
	auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
	//auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
	auto material_left = make_shared<dielectric>(1.5);
	//auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8));
	auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);
	//auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2));

	//auto R = cos(pi / 4);
	//auto material_blue = make_shared<lambertian>(color(0, 0, 1));
	//auto material_red = make_shared<lambertian>(color(1, 0, 0));

	//world.add(make_shared<Sphere>(R, point3(-R, 0, -1), material_blue));
	//world.add(make_shared<Sphere>(R, point3(R, 0, -1), material_red));

	world.add(make_shared<Sphere>(0.5, point3(0, 0, -1), material_center));
	world.add(make_shared<Sphere>(100, point3(0, -100.5, -1), material_ground));
    world.add(make_shared<Sphere>(-0.45, point3(-1.0, 0, -1), material_left));   //normals become inwards
	world.add(make_shared<Sphere>(0.5, point3(-1.0, 0, -1), material_left));
	world.add(make_shared<Sphere>(0.5, point3(1.0, 0, -1), material_right));


	//Camera camera
	point3 lookfrom(3, 3, 2);
	point3 lookat(0, 0, -1);
	Vector3 vup(0, 1, 0);
	auto dist_to_focus = (lookfrom - lookat).length();
	auto aperture = 2.0;
	//std::cerr << "Dist: " << dist_to_focus << std::endl;
	camera cam(lookfrom, lookat, vup,  20.0, aspect_ratio, aperture, dist_to_focus);   //90.0 is the correct fov for 12.2
	
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
				ray r = cam.get_ray(u, v);  r.m_depth = max_depth;
				pixel_color += ray_color_world(r, world);
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
}*/

