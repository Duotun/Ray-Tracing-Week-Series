#pragma once
#ifndef PERLIN_H
#define PERLIN_H

#include "utility.hpp"

// create perlin noise for 3D Cases
// the core ideas for the perlin is using random generator to permute the sequences of numbers
class perlin {

public:
	perlin() {
		ranfloat = new double[point_count];
		for (int i = 0; i < point_count; i++)
		{
			ranfloat[i] = random_double();
		}

		perm_x = perlin_generate_perm();
		perm_y = perlin_generate_perm();
		perm_z = perlin_generate_perm();
	}

	~perlin()
	{
		delete[] ranfloat;
		delete[] perm_x;
		delete[] perm_y;
		delete[] perm_z;
	}

	double noise(const point3& p) const   //return the noise value from ranfloat array
	{
		auto i = static_cast<int>(4 * p.x()) & 255;
		auto j = static_cast<int>(4 * p.y()) & 255;
		auto k = static_cast<int>(4 * p.z()) & 255;

		return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];  //perform xor operations
	}
private:
	static const int point_count = 256;
	double* ranfloat;
	int* perm_x;
	int* perm_y;
	int* perm_z;

	static int* perlin_generate_perm()
	{
		auto p = new int[point_count];
		for (int i = 0; i < perlin::point_count; i++)
		{
			p[i] = i;
		}

		return p;
	}

	static void permute(int* p, int n)
	{
		//perform the random permutation
		for (int i = n - 1; i > 0; i--)
		{
			int target = random_int(0, i);
			std::swap(p[i], p[target]);
		}
	}
};

#endif // ! PERLIN_H

