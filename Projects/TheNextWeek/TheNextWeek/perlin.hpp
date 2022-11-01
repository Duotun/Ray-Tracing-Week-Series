#pragma once
#ifndef PERLIN_H
#define PERLIN_H

#include "utility.hpp"

// create perlin noise for 3D Cases
// the core ideas for the perlin is using random generator to permute the sequences of numbers
class perlin {

public:
	perlin() {
		ranvec = new Vector3[point_count];
		for (int i = 0; i < point_count; i++)
		{
			ranvec[i] = unit_vector(Vector3::random(-1.0, 1.0));   //initialize the ran vec unit vectors
		}

		perm_x = perlin_generate_perm();
		perm_y = perlin_generate_perm();
		perm_z = perlin_generate_perm();
	}

	~perlin()
	{
		delete[] ranvec;
		delete[] perm_x;
		delete[] perm_y;
		delete[] perm_z;
	}

	double noise(const point3& p) const   //return the noise value from ranfloat array
	{
		// try smoothing with interpolation
		auto u = p.x() - floor(p.x());
		auto v = p.y() - floor(p.y());
		auto w = p.z() - floor(p.z());    // the factor for performing interpolation [0, 1]


		auto i = static_cast<int>(floor(p.x()));
		auto j = static_cast<int>(floor(p.y()));
		auto k = static_cast<int>(floor(p.z()));

		Vector3 c[2][2][2];  //save the boundary values for interpolation
		for(int di=0; di<2; di++)
			for(int dj=0; dj<2; dj++)
				for (int dk = 0; dk < 2; dk++)
				{
					c[di][dj][dk] = ranvec[perm_x[(i + di)&255] ^ perm_y[(j + dj)&255] ^ perm_z[(k + dk)&255]];
				}


		return trilinear_interp(c, u, v, w);  //perform xor operations
	}

	double turb(const point3& p, int depth = 7) const {   //use multiple summed frequencies 
		auto accum = 0.0;
		point3 temp_p = p;
		auto weight = 1.0;

		for (int i = 0; i < depth; i++)
		{
			accum += weight * noise(temp_p);
			weight *= 0.5;
			temp_p *= 2.0;
		}

		return fabs(accum);  //fabs is in <cmath>
	}
private:
	static const int point_count = 256;
	Vector3* ranvec;  //use vector rather than floats to solve block looking like
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

	// right now the noise value is between [-1, 1]
	static double trilinear_interp(Vector3 c[2][2][2], double u, double v, double w)
	{
		// add hermitan smoothing 
		auto uu = u * u * (3.0 - 2 * u);
		auto vv = v * v * (3.0 - 2 * v);
		auto ww = w * w * (3.0 - 2 * w);
		
		auto accum = 0.0;

		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 2; k++) {
					Vector3 weight_v(u - i, v - j, w - k);
					accum += (i * uu + (1 - i) * (1 - uu))
						* (j * vv + (1 - j) * (1 - vv))
						* (k * ww + (1 - k) * (1 - ww))
						* dot(c[i][j][k], weight_v);
				}

		return accum;
	}
};

#endif // ! PERLIN_H

