
#pragma region
//#includes
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

#include "utility.hpp"
#include "ray.hpp"
#include "material.hpp"
//#include "pdf.hpp"
#include "hittable.hpp"
#include "hittablelist.hpp"
#include "aarect.hpp"
#include "box.hpp"
#include "camera.hpp"
#include "color.hpp"
#include "sphere.hpp"

#pragma endregion


//world drawing
color ray_color_world(ray& r, const color& background, const hittable& world,
    shared_ptr<hittable> lights)
{
    hit_record rec;
    if (r.m_depth <= 0)  // totally no hit after recursions
    {
        return color(0, 0, 0);
    }

    if (!world.Intersect(r, rec))  // background for no hitting of this layer
        return background;

    ray scattered; color attenuation;
    scatter_record srec;
    color emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
    //double pdf_val;
    if (!rec.mat_ptr->scatter(r, rec, srec))
        return emitted;   // if emission materials only

    if (srec.is_specular) {
        srec.specular_ray.m_depth = r.m_depth - 1;
        return srec.attenuation
            * ray_color_world(srec.specular_ray, background, world, lights);
    }
    auto light_ptr = make_shared<hittable_pdf>(lights, rec.p); //for lighting pdf
    mixture_pdf mixed_pdf(light_ptr, srec.pdf_ptr);

    scattered = ray(rec.p, mixed_pdf.generate(), r.time());
    auto pdf_val = mixed_pdf.value(scattered.direction());
    scattered.m_depth = r.m_depth - 1;
    scattered.m_tmin = 0.001; scattered.m_tmax = infinity;
    return emitted + srec.attenuation * rec.mat_ptr->scattering_pdf(r, rec, scattered) *
        ray_color_world(scattered, background, world, lights)/pdf_val;
}

// the cornell_box for light testing
hittable_list cornell_box() {
    hittable_list objects;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<flip_face>(make_shared<xz_rect>(213, 343, 227, 332, 554, light)));

    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    shared_ptr<material> aluminum = make_shared<metal>(color(0.8, 0.85, 0.88), 0.0);
    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);  //RTS Sequence
    box1 = make_shared<translate>(box1, Vector3(265, 0, 295));
    objects.add(box1);

    //shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
    //box2 = make_shared<rotate_y>(box2, -18);
    //box2 = make_shared<translate>(box2, Vector3(130, 0, 65));
    //objects.add(box2);

    auto glass = make_shared<dielectric>(1.5);
    objects.add(make_shared<sphere>(point3(190, 90, 190), 90, glass));

    return objects;
}
int main()
{
	//image
	auto aspect_ratio = 1.0;
	int image_width = 600;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	int samples_per_pixel = 100;
	int max_depth = 50;

	//World
	auto world = cornell_box();
    auto lights = make_shared<hittable_list>();
    lights->add(make_shared<xz_rect>(213, 343, 227, 332, 554, shared_ptr<material>()));
    lights->add(make_shared<sphere>(point3(190, 90, 190), 90, shared_ptr<material>()));
    
    color background(0, 0, 0);

    //camera 
    aspect_ratio = 1.0;
    image_width = 600;
    samples_per_pixel = 200;
    background = color(0, 0, 0);
    point3 lookfrom = point3(278, 278, -800);
    point3 lookat = point3(278, 278, 0);
    auto vfov = 40.0;
    Vector3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0;

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    image_height = static_cast<int>(image_width / aspect_ratio);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    //wirte the image from left to right, top to bottom (first row to the last row)
    //so pmm still defines bottom-left at (0, 0)
    for (int i = image_height - 1; i >= 0; --i)  // bottom becomes black
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int j = 0; j < image_width; ++j)
        {

            //double b = 0.25;
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; s++) {
                auto u = double(j + random_double()) / (image_width - 1.0);
                auto v = double(i + random_double()) / (image_height - 1.0);
                ray r = cam.get_ray(u, v);  r.m_depth = max_depth;
                pixel_color += ray_color_world(r, background, world, lights);
            }


            //simple ppm format, tone map to 0-255 + anti-aliasing
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
    std::cerr << "\nDone.\n";
	return 0;
}