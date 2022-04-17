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
#pragma endregion

class Vector3 {
	public:
		double p[3];
		constexpr Vector3() noexcept : p{0, 0, 0} {}
		constexpr Vector3(double x, double y, double z) : p{ x, y, z } {}
		Vector3(const Vector3& v) {
			p[0] = v.p[0];
			p[1] = v.p[1];
			p[2] = v.p[2];
		}
		constexpr Vector3(Vector3&& v) = default;
		~Vector3() = default;


		//assignment operators
		Vector3& operator=(const Vector3& v) = default;
		Vector3& operator=(Vector3 && v) = default;

		//access the pos elements
		double x() const { return p[0];}
		double y() const { return p[1];}
		double z() const { return p[2];}

		//obtain elements from index
		Vector3 operator-() const { return Vector3(-p[0], -p[1], -p[2]); }
		double operator[](int i) const { return  p[i]; }
		double& operator[](int i)  { return  p[i]; }

		// basic operators
		Vector3& operator+=(const Vector3& v)
		{
			p[0] += v.p[0];
			p[1] += v.p[1];
			p[2] += v.p[2];
			return *this;
		}

		Vector3& operator+=(const double t)
		{
			p[0] += t;
			p[1] += t;
			p[2] += t;
			return *this;
		}

		Vector3& operator-=(const Vector3& v)
		{
			p[0] -= v.p[0];
			p[1] -= v.p[1];
			p[2] -= v.p[2];
			return *this;
		}

		Vector3& operator*=(const Vector3& v)
		{
			p[0] *= v.p[0];
			p[1] *= v.p[1];
			p[2] *= v.p[2];
			return *this;
		}

		Vector3& operator/=(const Vector3& v)
		{
			p[0] /= v.p[0];
			p[1] /= v.p[1];
			p[2] /= v.p[2];
			return *this;
		}

		[[nodiscard]]
		constexpr Vector3 operator+(const Vector3& v) const noexcept {
			return { p[0]+ v.p[0], p[1] + v.p[1], p[2] + v.p[2] };
		}

		[[nodiscard]]
		constexpr  Vector3 operator-(const Vector3& v) const noexcept {
			return { p[0] - v.p[0], p[1] - v.p[1], p[2] - v.p[2] };
		}

		[[nodiscard]]
		constexpr  Vector3 operator*(const Vector3& v) const noexcept {
			return { p[0] * v.p[0], p[1] * v.p[1], p[2] * v.p[2] };
		}

		[[nodiscard]]
		constexpr  Vector3 operator/(const Vector3& v) const noexcept {
			return { p[0] / v.p[0], p[1] / v.p[1], p[2] / v.p[2] };
		}

		[[nodiscard]]
		constexpr Vector3 operator+(double a) const noexcept {
			return { p[0] + a, p[1] + a, p[2] + a };
		}

		[[nodiscard]]
		constexpr  Vector3 operator-(double a) const noexcept {
			return { p[0] - a, p[1] - a, p[2] - a };
		}

		[[nodiscard]]
		constexpr  Vector3 operator*(double a) const noexcept {
			return { p[0] * a, p[1] * a, p[2] * a };
		}

		[[nodiscard]]
		constexpr Vector3 operator/(double a) const noexcept {
			const double inv_a = 1.0 / a;
			return { p[0] * inv_a, p[1] * inv_a, p[2] * inv_a };
		}

		double length_squared() const noexcept
		{
			return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
		}

		double length() const noexcept {
			return sqrt(length_squared());
		}

		void normalize() {
			*this / length();
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

inline Vector3 operator*(const Vector3& v, double t) {
	return Vector3{ v.p[0] * t, v.p[1] * t, v.p[2] * t };
}

inline Vector3 operator*(double t, const Vector3& v) {
	return Vector3{ v.p[0] * t, v.p[1] * t, v.p[2] * t };
}


constexpr double dot(const Vector3& u, const Vector3& v)
{
	return u.p[0] * v.p[0] + u.p[1] * v.p[1] + u.p[2] * v.p[2];
}

constexpr Vector3 cross(const Vector3& u, const Vector3& v)
{
	return Vector3{u.p[1] * v.p[2] - u.p[2] * v.p[1],
	u.p[2] * v.p[0] - u.p[0] * v.p[2],
	u.p[0] * v.p[1] - u.p[1] * v.p[0] };
}

constexpr Vector3 unit_vector(Vector3 v)
{
	return v / v.length();
}
