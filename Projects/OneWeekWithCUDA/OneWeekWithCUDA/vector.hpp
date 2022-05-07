#pragma once
#ifndef VECTOR_H   //avoid multiple include of class
#define VECTOR_H

#pragma region
//includes
#include<cmath>
#include<iostream>
#include<algorithm>
#include <cstdint>
#include <limits>
#include "device_launch_parameters.h"
#include "utility.hpp"
#pragma endregion

class Vector3 {
public:
	float p[3];
	__host__ __device__ Vector3() noexcept : p{ 0, 0, 0 } {}   //support both gpu and cpu parts
	__host__ __device__ Vector3(float x, float y, float z) : p{ x, y, z } {}
	__host__ __device__ Vector3(const Vector3& v) {
		p[0] = v.p[0];
		p[1] = v.p[1];
		p[2] = v.p[2];
	}
	__host__ __device__ Vector3(Vector3&& v) = default;
	__host__ __device__ ~Vector3() = default;


	//assignment operators
	__host__ __device__ Vector3& operator=(const Vector3& v) = default;
	__host__ __device__ Vector3& operator=(Vector3&& v) = default;

	//access the pos elements
	__host__ __device__ float x() const { return p[0]; }
	__host__ __device__ float y() const { return p[1]; }
	__host__ __device__ float z() const { return p[2]; }

	//obtain elements from index
	__host__ __device__ Vector3 operator-() const { return Vector3(-p[0], -p[1], -p[2]); }
	__host__ __device__ float operator[](int i) const { return  p[i]; }
	__host__ __device__ float& operator[](int i) { return  p[i]; }

	// basic operators
	__host__ __device__ Vector3& operator+=(const Vector3& v)
	{
		p[0] += v.p[0];
		p[1] += v.p[1];
		p[2] += v.p[2];
		return *this;
	}

	__host__ __device__ Vector3& operator+=(const float t)   //coercion conversion if passed type is double
	{
		p[0] += t;
		p[1] += t;
		p[2] += t;
		return *this;
	}

	__host__ __device__ Vector3& operator-=(const Vector3& v)  
	{
		p[0] -= v.p[0];
		p[1] -= v.p[1];
		p[2] -= v.p[2];
		return *this;
	}

	__host__ __device__ Vector3& operator*=(const Vector3& v)
	{
		p[0] *= v.p[0];
		p[1] *= v.p[1];
		p[2] *= v.p[2];
		return *this;
	}

	__host__ __device__ Vector3& operator/=(const Vector3& v)
	{
		p[0] /= v.p[0];
		p[1] /= v.p[1];
		p[2] /= v.p[2];
		return *this;
	}

	[[nodiscard]]
	__host__ __device__ Vector3 operator+(const Vector3& v) const noexcept {
		return { p[0] + v.p[0], p[1] + v.p[1], p[2] + v.p[2] };
	}

	[[nodiscard]]
	__host__ __device__ Vector3 operator-(const Vector3& v) const noexcept {
		return { p[0] - v.p[0], p[1] - v.p[1], p[2] - v.p[2] };
	}

	[[nodiscard]]
	__host__ __device__ Vector3 operator*(const Vector3& v) const noexcept {
		return { p[0] * v.p[0], p[1] * v.p[1], p[2] * v.p[2] };
	}

	[[nodiscard]]
	__host__ __device__ Vector3 operator/(const Vector3& v) const noexcept {
		return { p[0] / v.p[0], p[1] / v.p[1], p[2] / v.p[2] };
	}

	[[nodiscard]]
	__host__ __device__ Vector3 operator+(float a) const noexcept {
		return { p[0] + a, p[1] + a, p[2] + a };
	}


	[[nodiscard]]
	__host__ __device__ Vector3 operator*(float a) const noexcept {
		return { p[0] * a, p[1] * a, p[2] * a };
	}


	[[nodiscard]]
	__host__ __device__ Vector3 operator/(float a) const noexcept {
		const float inv_a = 1.0f / a;
		return { p[0] * inv_a, p[1] * inv_a, p[2] * inv_a };
	}


	[[nodiscard]]
	__host__ __device__ float length_squared() const noexcept
	{
		return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
	}

	[[nodiscard]]
	__host__ __device__ float length() const noexcept {
		return sqrt(length_squared());
	}

	__host__ __device__ void normalize() {
		p[0] /= length();
		p[1] /= length();
		p[2] /= length();
	}


	__host__ __device__ bool near_zero() const {
		const auto epsilon = 1e-8;
		return (fabs(p[0]) < epsilon) && (fabs(p[1]) < epsilon) && (fabs(p[2]) < epsilon);
	}

};

using point3 = Vector3;
using color = Vector3;   // to represent RGB, no alpha channel
#endif // !VECTOR_H

//Vector3 utility functions

inline std::ostream& operator<<(std::ostream& out, const Vector3& v)
{
	return out << v.p[0] << ' ' << v.p[1] << ' ' << v.p[2];
}

inline std::istream& operator>>(std::istream& is, Vector3& t) {
	is >> t.p[0] >> t.p[1] >> t.p[2];
	return is;
}


__device__ __host__ inline Vector3 operator*(const Vector3& v, float t) {
	return Vector3{ v.p[0] * t, v.p[1] * t, v.p[2] * t };
}

__device__ __host__ inline Vector3 operator*(float t, const Vector3& v) {
	return Vector3{ v.p[0] * t, v.p[1] * t, v.p[2] * t };
}


__device__ __host__ double dot(const Vector3& u, const Vector3& v)
{
	return (double)u.p[0] * v.p[0] + (double)u.p[1] * v.p[1] + (double)u.p[2] * v.p[2];
}

__device__ __host__ Vector3 cross(const Vector3& u, const Vector3& v)
{
	return Vector3{ u.p[1] * v.p[2] - u.p[2] * v.p[1],
	u.p[2] * v.p[0] - u.p[0] * v.p[2],
	u.p[0] * v.p[1] - u.p[1] * v.p[0] };
}

__device__ __host__ Vector3 unit_vector(Vector3 v)
{
	return v / v.length();
}


__device__ __host__ Vector3 reflect(const Vector3& v, const Vector3& n)
{
	return v - 2.0f * dot(v, n) * n;
}

__device__ __host__ Vector3 refract(const Vector3& uv, const Vector3& n, float etai_over_etat)
{
	float cos_theta = fmin(float(dot(-uv, n)), 1.0f);
	Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	Vector3 r_out_parallel = -sqrt(1.0f - r_out_perp.length_squared()) * n;
	return r_out_perp + r_out_parallel;
}

#define RANDVEC3 Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
__device__ Vector3 random_unit_sphere(curandState* local_rand_state)
{
	Vector3 p;
	do {
		p = 2.0f * RANDVEC3 - Vector3(1, 1, 1);
		//p = Vector3(random_curand_range(local_rand_state, -1.0f, 1.0f), random_curand_range(local_rand_state, -1.0f, 1.0f), random_curand_range(local_rand_state, -1.0f, 1.0f));
	} while (p.length_squared() >= 1.0f);

	return p;
}

