#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H

#include "utility.hpp"
#include "rtw_stb_image.hpp"
#include  <iostream>
#include "perlin.hpp"
//we combined texture and color together
class texture {
public:
	virtual color value(double u=0.0, double v=0.0, const point3& p = point3(0.0,0.0,0.0)) const = 0;
};

class solid_color : public texture {
public:
	solid_color() {
		color_value = color(1.0, 1.0, 1.0);
	}
	solid_color(color& c) : color_value(c){}
	solid_color(const color &c) : color_value(c){}
	solid_color(double red, double green, double blue) : 
		solid_color(color(red, green, blue)) {}
	~solid_color() noexcept = default;

	virtual color value(double u, double v, const point3& p) const override {
		return color_value;
	}
	//members
private:
	color color_value;
};

class checker_texture : public texture {

	//even - one color, odd - another color
public:
	checker_texture(){}
	checker_texture(color c1, color c2) : even(make_shared<solid_color>(c1)), odd(make_shared<solid_color>(c2)) {}

	virtual color value(double u, double v, const point3& p) const override {
		auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}

	shared_ptr<texture> odd;
	shared_ptr<texture>even;

};

class image_texture : public texture {
public:
	const static int bytes_per_pixel = 3; // 3 channels
	image_texture()
		: data(nullptr), width(0), height(0), bytes_per_scanline(0){}
	image_texture(const char* filename)
	{
		//stb_load into channels of value [0, 255]
		int components_per_pixel = bytes_per_pixel;
		width = height = 0;
		data = stbi_load(filename, &width, &height, &components_per_pixel, components_per_pixel);
		if (!data)
		{
			std::cerr << "Error: Could not load texture image file " << filename << " .\n";
			width = height = 0;
		}

		bytes_per_scanline = width * bytes_per_pixel;

	}
	~image_texture() { delete[] data; }

	virtual color value(double u, double v, const point3& p) const override {
		//if no data return a solid cyan
		if (data == nullptr)
			return color(0, 1, 1);

		u = std::clamp(u, 0.0, 1.0);       //get the uv and map to the image
		v = 1.0 - std::clamp(v, 0.0, 1.0);   //flip v to image coordinates

		auto i = static_cast<int> (u * width);
		auto j = static_cast<int> (v * height);

		const auto color_scale = 1.0 / 255.0;  //make the image reading back to [0, 1
		auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;
		return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
	}
private:
	unsigned char* data;
	int width, height;
	int bytes_per_scanline;
};

class noise_texture : public texture {
public:
	noise_texture(){}

	virtual color value(double u, double v, const point3& p) const override
	{
		return color(1, 1, 1) * noise.noise(p);
	}

    //member, using perlin noise
	perlin noise;
};
#endif // !TEXTURE_H
