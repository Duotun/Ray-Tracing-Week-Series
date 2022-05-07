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
class Sphere : public hittable {

public:
	double m_r;
	Vector3 m_p;  //center position
	shared_ptr<material> mat_ptr;

	//Methods
	explicit Sphere(double r,
		Vector3 p,
		shared_ptr<material> m) noexcept
		: m_r(r),
		m_p(std::move(p)),
		mat_ptr(m) {}
	Sphere(const Sphere& sphere) noexcept = default;
	Sphere(Sphere&& sphere) noexcept = default;
	~Sphere() = default;


	//---------------------------------------------------------------------
	// Assignment Operators
	//---------------------------------------------------------------------

	Sphere& operator=(const Sphere& sphere) = default;
	Sphere& operator=(Sphere&& sphere) = default;

	//bouding box
	
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
	//intersection method
	[[nodiscard]]
	bool Intersect(ray& r, hit_record& rec) const noexcept {  //need to update the r and hit records
		const Vector3 op = m_p - r.org;
		const double dop = dot(unit_vector(r.dir), op);
		const double D = dop * dop - dot(op, op) + m_r * m_r;

		if (0.0 > D) {   //D is determinant
			return false;
		}

		const double sqrtD = sqrt(D);
		double tmin = dop - sqrtD;

		//no intersection at all, two results are not in the range
		if (tmin <= r.m_tmin || tmin >= r.m_tmax)
		{
			tmin = dop + sqrtD;  //test the second result
			if (tmin <= r.m_tmin || tmin >= r.m_tmax)
				return false;

		}

		//r.m_tmax = tmin;  //update m_tmax in the hittable_list
		rec.t = tmin;
		rec.p = r.at(rec.t);
		Vector3 outward_normal = unit_vector((rec.p - m_p) / m_r);
		//rec.normal = outward_normal;
		rec.set_face_normal(r, outward_normal);   //normalized normal
		get_sphere_uv(outward_normal, rec.u, rec.v);
		rec.mat_ptr = mat_ptr;  // use pointer to pass values
		return true;
	}

private:
	static void get_sphere_uv(const point3& p, double& u, double& v)
	{
		//p agiven point on the sphere of radius one, centered at the origin (normal)
		auto theta = acos(-p.y());
		auto phi = atan2(-p.z(), p.x()) + pi;

		u = phi / (2 * pi);
		v = theta / pi;  //map to [0, 1]
	}
	
};

bool Sphere::bounding_box(double time0, double time1, aabb& output_box) const
{
	output_box = aabb(
		m_p - Vector3(m_r, m_r, m_r),
		m_p - Vector3(m_r, m_r, m_r)
	);
	return true;
}
#endif // !SPHERE_H