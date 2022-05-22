#pragma once
#ifndef COLOR_H
#define COLOR_H

#pragma region
//includes
#include "vector.hpp"
#include<iostream>
#include<cstdint>
#include<limits>
#include <algorithm>
#pragma endregion

//define a function for write the translated [0, 255] values of each color component
// pixel_color are recongnized in the range (0, 1)

const double gamma = 2;  //2.2 is better, while we use 2 to correspond to the tutorial
void write_color(std::ostream& out, color pixel_color)
{
	//apply simple gamma correction here
	pixel_color.p[0] = std::pow(pixel_color.x(), 1.0 / gamma);
	pixel_color.p[1] = std::pow(pixel_color.y(), 1.0 / gamma);
	pixel_color.p[2] = std::pow(pixel_color.z(), 1.0 / gamma);

	out << static_cast<std::uint32_t>(std::clamp(255.999 * pixel_color.x(), 0.0, 255.0)) << ' '
		<< static_cast<std::uint32_t>(std::clamp(255.999 * pixel_color.y(), 0.0, 255.0)) << ' '
		<< static_cast<std::uint32_t>(std::clamp(255.999 * pixel_color.z(), 0.0, 255.0)) << '\n';
}

void write_color(std::ostream& out, color pixel_color, int samples_per_pixel)
{

	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	if (r != r) r = 0.0;
	if (g != g) g = 0.0;
	if (b != b) b = 0.0;   //for the unwanted NaN cases

	//divide the color by the number of samplers
	auto scale = 1.0 / samples_per_pixel;
	r *= scale;  g *= scale; b *= scale;
	//add simple gamma correction here for each channel
	r = std::pow(r, 1.0 / gamma);
	g = std::pow(g, 1.0 / gamma);
	b = std::pow(b, 1.0 / gamma);

	out << static_cast<std::uint32_t>(std::clamp(255.999 * r, 0.0, 255.0)) << ' '
		<< static_cast<std::uint32_t>(std::clamp(255.999 * g, 0.0, 255.0)) << ' '
		<< static_cast<std::uint32_t>(std::clamp(255.999 * b, 0.0, 255.0)) << '\n';
}

#endif // !COLOR_H
