#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

//includes
#include "utility.hpp"
#include "hittable.hpp"
#include "hit_record.hpp"
#include "texture.hpp"
#include "onb.hpp"

class hit_record;

//define a scatter record to help differentiate specular and diffuse material
struct scatter_record {
	ray specular_ray;
	bool is_specular;
	color attenuation;
	shared_ptr<pdf>pdf_ptr;
};

//produce a scattered ray and define how much the ray should be attenuated
class material {
public:
	virtual color emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p) const {
		return color(0, 0, 0);  //default no emission
	}
	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, double & pdf) const {
		return false;
	}

	virtual double scattering_pdf(
		const ray& r_in, const hit_record& rec, const ray& scattered
	) const {
		return 0;
	}
};



class lambertian : public material {
public:
	shared_ptr<texture> albedo;

public:
	lambertian(const color& a) :albedo(make_shared<solid_color>(a)) {}
	lambertian(shared_ptr<texture> a) : albedo(a) {}
	lambertian() :albedo(make_shared<solid_color>(color{ 1.0, 1.0, 1.0 })) {}  //default is the white

	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, double &pdf) const override {
		onb uvw;
		uvw.build_from_w(rec.normal);
		auto direction = uvw.local(random_cosine_direction());
		scattered = ray(rec.p, direction, r_in.time());  //generate the scattered ray
		attenuation = albedo->value(rec.u, rec.v, rec.p); // and it is a solid color
		pdf = dot(uvw.w(), scattered.dir) / pi;   // compute pdf
		return true;   //true only
	}

	double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
		const {
		auto cosine = dot(rec.normal, unit_vector(scattered.dir));
		return cosine < 0 ? 0.0 : cosine / pi;
	}

};


class metal : public material {
public:
	color albedo;
	double fuzz;
public:
	//add fuzzy disturb for the reflected ray
	metal(const color& a, double f = 0.0) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}   // fuzz = zero equals no disturbance
	metal() :albedo({ 1.0, 1.0, 1.0 }), fuzz(0.0) {}  //default is the white, no fuzz
	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, double& pdf) const override {
		Vector3 reflected = reflect(unit_vector(r_in.dir), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(), r_in.time());
		attenuation = albedo;
		return (dot(scattered.dir, rec.normal) > 0);   //outside the surface
	}

};

class dielectric : public material {
public:
	double ir; //index of Refraction
public:
	dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, double &pdf) const override {
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		Vector3 unit_direction = unit_vector(r_in.dir);
		//need to care about the internal refraction which is total reflect case
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
		Vector3 dir;
		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {   //total reflect, if looking at an angle
			dir = reflect(unit_direction, rec.normal);
		}
		else
		{
			dir = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = ray(rec.p, dir, r_in.time());
		return true;   //glass absorbs nothing
	}
private:
	static double reflectance(double cosine, double ref_idx)
	{
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};


//emission materials
class diffuse_light : public material {

public:
	diffuse_light(shared_ptr<texture> a) : emit(a) {}
	diffuse_light(color c) : emit(make_shared<solid_color>(c)) {}

	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, double &pdf) const override {
		return false;  //no reflection, no attenuation
	}

	virtual color emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p) const override {
		if (rec.front_face)   //light from only one direction
			return emit->value(u, v, p);
		else
			return color(0, 0, 0);
	}

	//members
	shared_ptr<texture> emit;
};

//for volume 
class isotropic : public material {
public:
	isotropic(color c) :albedo(make_shared<solid_color>(c)) {}
	isotropic(shared_ptr<texture> a) : albedo(a) {}

	virtual bool scatter(const ray& r_in, hit_record& rec, color& attenuation, ray& scattered, double &pdf) const override {
		// just return a random scatter ray
		scattered = ray(rec.p, random_in_unit_sphere(), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	//texture member
	shared_ptr<texture>albedo;
};
#endif //  MATERIAL_H
