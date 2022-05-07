#pragma region
//includes
#include <iostream>
#include "utility.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "moving_sphere.hpp"
#include "hittablelist.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "aarect.hpp"
#include "box.hpp"
#include "constant_medium.hpp"
#include "bvh.hpp"
#pragma endregion


hittable_list final_scene() {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1, 101);
            auto z1 = z0 + w;

            boxes1.add(make_shared<box>(point3(x0, y0, z0), point3(x1, y1, z1), ground));
        }
    }

    hittable_list objects;

    objects.add(make_shared<bvh_node>(boxes1, 0, 1));

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + Vector3(30, 0, 0);
    auto moving_sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50, moving_sphere_material));

    objects.add(make_shared<Sphere>(50, point3(260, 150, 45), make_shared<dielectric>(1.5)));
    objects.add(make_shared<Sphere>(50, point3(0, 150, 145), make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
        ));

    auto boundary = make_shared<Sphere>(70, point3(360, 150, 145), make_shared<dielectric>(1.5));
    objects.add(boundary);
    objects.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = make_shared<Sphere>(5000, point3(0, 0, 0), make_shared<dielectric>(1.5));
    objects.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

    auto emat = make_shared<lambertian>(make_shared<image_texture>("../earthmap.jpg"));
    objects.add(make_shared<Sphere>(100, point3(400, 200, 400), emat));
    //auto pertext = make_shared<noise_texture>(0.1);
    //objects.add(make_shared<sphere>(point3(220, 280, 300), 80, make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<Sphere>(10, point3::random(0, 165), white));
    }

    objects.add(make_shared<translate>(
        make_shared<rotate_y>(
            make_shared<bvh_node>(boxes2, 0.0, 1.0), 15),
        Vector3(-100, 270, 395)
        )
    );

    return objects;
}
hittable_list cornell_smoke() {
    hittable_list objects;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(7, 7, 7));

    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<xz_rect>(113, 443, 127, 432, 554, light));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, Vector3(265, 0, 295));

    shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, Vector3(130, 0, 65));

    objects.add(make_shared<constant_medium>(box1, 0.01, color(0, 0, 0)));
    objects.add(make_shared<constant_medium>(box2, 0.01, color(1, 1, 1)));

    return objects;
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
    objects.add(make_shared<xz_rect>(213, 343, 227, 332, 554, light));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    //add two boxes
    //objects.add(make_shared<box>(point3(130, 0, 65), point3(295, 165, 230), white));
    //objects.add(make_shared<box>(point3(265, 0, 295), point3(430, 330, 460), white));
    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);  //RTS Sequence
    box1 = make_shared<translate>(box1, Vector3(265, 0, 295));
    objects.add(box1);

    shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, Vector3(130, 0, 65));
    objects.add(box2);
   
    return objects;
}
// scene collections
hittable_list earth()
{
    auto earth_texture = make_shared<image_texture>(".../earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<Sphere>(2, point3(0, 0, 0), earth_surface);

    return hittable_list(globe);

}

hittable_list simple_light()
{
    hittable_list objects;
    auto scolor = make_shared<solid_color>(0.73, 0.73, 0.73);
    objects.add(make_shared<Sphere>(1000, point3(0, -1000, 0), make_shared<lambertian>(scolor)));
    objects.add(make_shared<Sphere>(2, point3(0, 2, 0), make_shared<lambertian>(scolor)));

    auto diffuselight = make_shared<diffuse_light>(color(4, 4, 4));
    //objects.add(make_shared<Sphere>(2, point3(0, 7, 0), diffuselight));
    objects.add(make_shared<xy_rect>(3, 5, 1, 3, -2, diffuselight));

    return objects;
}
hittable_list two_spheres()
{
    hittable_list objects;
    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

    objects.add(make_shared<Sphere>(10, point3(0, -10, 0), make_shared<lambertian>(checker)));
    objects.add(make_shared<Sphere>(10, point3(0, 10, 0), make_shared<lambertian>(checker)));

    return objects;
}


//world drawing
color ray_color_world(ray& r, const color& background, const hittable& world)
{
    hit_record rec;
    if (r.m_depth <= 0)  // totally no hit after recursions
    {
        //std::cerr << "Depth: " << r.m_depth << std::endl;
        return color(0, 0, 0);
    }

    r.m_tmin = 0.0001;
    r.m_tmax = infinity;
    //if(r.m_depth < 50)
    //    std::cerr << "Depth: " << r.m_depth << std::endl;
    // if hit nothing, return the background color
    if (!world.Intersect(r, rec))  // background for no hitting of this layer
        return background;
    
    ray scattered; color attenuation;
    color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
    if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
        return emitted;   // if emission materials only

    scattered.m_depth = r.m_depth - 1;
    scattered.m_tmin = 0.0001; scattered.m_tmax = infinity;
    return emitted + attenuation * ray_color_world(scattered, background, world);
    
    // gradient backgound
    ///Vector3 unit_direction = unit_vector(r.direction());
    ///auto t = 0.5 * (unit_direction.y() + 1.0);
    //return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    int samples_per_pixel = 100;
    const int max_depth = 50;


    // world
    hittable_list world;
    //Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    Vector3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0;
    auto vfov = 40.0;
    color background(0, 0, 0);   //color background

    //swtich case for scene choice
    switch (0)
    {
    case 1:
        //world = random_scene();
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vfov = 20.0;
        background = color(0.70, 0.80, 1.00);
        aperture = 0.1; break;

    case 2: world = two_spheres();
        background = color(0.70, 0.80, 1.00);
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vfov = 20.0;
        aperture = 0.1;
        break;
    //default:
    case 4:
        world = earth();
        background = color(0.70, 0.80, 1.00);
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vfov = 20.0;
        break;

    //default:
    case 5:
        background = color(0.0, 0.0, 0.0);
        world = simple_light();
        samples_per_pixel = 200;
        background = color(0, 0, 0);
        lookfrom = point3(26, 3, 6);
        lookat = point3(0, 2, 0);
        vfov = 20.0;
        break;

    //default:
    case 6:
        world = cornell_box();
        aspect_ratio = 1.0;
        image_width = 600;
        samples_per_pixel = 200;
        background = color(0, 0, 0);
        lookfrom = point3(278, 278, -800);
        lookat = point3(278, 278, 0);
        vfov = 40.0;
        break;

    //default:
    case 7:
        world = cornell_smoke();
        aspect_ratio = 1.0;
        image_width = 600;
        samples_per_pixel = 200;
        lookfrom = point3(278, 278, -800);
        lookat = point3(278, 278, 0);
        vfov = 40.0;
        break;

    default:
    case 8:
        world = final_scene();
        aspect_ratio = 1.0;
        image_width = 800;
        samples_per_pixel = 10000;
        background = color(0, 0, 0);
        lookfrom = point3(478, 278, -600);
        lookat = point3(278, 278, 0);
        vfov = 40.0;
        break;
    }

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    int image_height = static_cast<int>(image_width / aspect_ratio);  //change the height after aspect_ratio change

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
                pixel_color += ray_color_world(r, background, world);
            }


            //simple ppm format, tone map to 0-255 + anti-aliasing
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
    std::cerr << "\nDone.\n";
    return 0;
}