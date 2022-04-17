#pragma once
#ifndef SPHERE_H
#define SPHERE_H


#pragma region
//includes and defines
#include "ray.hpp"
#include <cstdint>
#include "hittable.hpp"
#define EPSILON_SPhere 1e-4    //used for intersection sanity check
#pragma endregion

enum struct Reflection_t : std::uint8_t {
	Diffuse = 0u,
	Specular,
	Reflective
};
class Sphere: public hittable{

public:
	double m_r;
	Vector3 m_p;  //center position
	Vector3 m_e;  //emission
	Vector3 m_f;  //reflection
	Reflection_t m_reflection_t = Reflection_t::Diffuse;
	constexpr explicit Sphere(double r,
		Vector3 p,
		Vector3 e = Vector3(0, 0, 0),
		Vector3 f = Vector3(0, 0, 0),
		Reflection_t reflection_t = Reflection_t::Diffuse) noexcept
		: m_r(r),
		m_p(std::move(p)),
		m_e(std::move(e)),
		m_f(std::move(f)),
		m_reflection_t(reflection_t) {}
	constexpr Sphere(const Sphere& sphere) noexcept = default;
	constexpr Sphere(Sphere && sphere) noexcept = default;
	~Sphere() = default;

	//---------------------------------------------------------------------
	// Assignment Operators
	//---------------------------------------------------------------------

	Sphere& operator=(const Sphere & sphere) = default;
	Sphere& operator=(Sphere && sphere) = default;

	//intersection method
	[[nodiscard]]
	bool Intersect(ray& r, hit_record& rec) const noexcept {  //need to update the r and hit records
		const Vector3 op = m_p - r.org;
		const double dop = dot(unit_vector(r.dir),op);
		const double D = dop * dop - dot(op,op) + m_r * m_r;

		if (0.0 > D) {   //D is determinant
			return false;
		}

		const double sqrtD = sqrt(D);
		double tmin = dop - sqrtD;
		
		//no intersection at all, two results are not in the range
		if (tmin<=r.m_tmin || tmin >= r.m_tmax)
		{
			tmin = dop + sqrtD;  //test the second result
			if (tmin <= r.m_tmin || tmin >= r.m_tmax)
				return false;
			
		}

		r.m_tmax = tmin;
		rec.t = r.m_tmax;
		rec.p = r.at(rec.t);
		Vector3 outward_normal = unit_vector((rec.p - m_p) / m_r);
		//rec.normal = outward_normal;
		rec.set_face_normal(r, outward_normal);   //normalized normal
		return true;
	}
};

#endif // !SPHERE_H