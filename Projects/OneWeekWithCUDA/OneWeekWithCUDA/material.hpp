#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H



class hit_record;
//includes
#include "utility.hpp"
#include "hitable.hpp"
#include "hit_record.hpp"


//produce a scattered ray and define how much the ray should be attenuated
class material {
public:
	__device__
	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};


class lambertian : public material {
public:
	color albedo;

public:
	__device__ lambertian(const color& a) :albedo(a) {}
	__device__ lambertian() :albedo({ 1.0, 1.0, 1.0 }) {}  //default is the white

	__device__ 
	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const  {
		Vector3 scatter_direction = rec.normal + random_unit_sphere(local_rand_state);
		
		//solve scatter direction is too bad
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

		scattered = ray(rec.p, scatter_direction);  //generate the scattered ray
		attenuation = albedo;
		return true;
	}

};


class metal : public material {
public:
	color albedo;
	double fuzz;
public:
	//add fuzzy disturb for the reflected ray
	__device__ metal(const color& a, double f = 0.0) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}   // fuzz = zero equals no disturbance
	__device__ metal() :albedo({ 1.0, 1.0, 1.0 }), fuzz(0.0) {}  //default is the white, no fuzz
	__device__ virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, 
		ray& scattered, curandState* local_rand_state) const {
		Vector3 reflected = reflect(unit_vector(r_in.dir), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.dir, rec.normal) > 0);   //outside the surface
	}

};

class dielectric : public material {
public:
	float ir; //index of Refraction
public:
	__device__ dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	__device__ virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, 
		ray& scattered, curandState* local_rand_state) const  {

		attenuation = color(1.0f, 1.0f, 1.0f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
		Vector3 unit_direction = unit_vector(r_in.dir);
		//need to care about the internal refraction which is total reflect case
		float cos_theta = fmin(float(dot(-unit_direction, rec.normal)), 1.0f);
		float sin_theta = sqrt(1.0f- cos_theta * cos_theta);
		Vector3 dir;
		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) >curand_uniform(local_rand_state)) {   //total reflect, if looking at an angle
			dir = reflect(unit_direction, rec.normal);
		}
		else
		{
			dir = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = ray(rec.p, dir);
		return true;   //glass absorbs nothing
	}
private:
	__device__ static float reflectance(float cosine, float ref_idx)
	{
		// Use Schlick's approximation for reflectance.
		auto r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
	}
};

#endif //  MATERIAL_H
