#pragma once
#ifndef COLOR_H
#define COLOR_H

#pragma region
//includes
#include "vector.hpp"
#include <iostream>
#include<cstdint>
#include<limits>
#pragma endregion

//define a function for write the translated [0, 255] values of each color component
// pixel_color are recongnized in the range (0, 1)

void write_color(std::ostream& out, color pixel_color)
{
	out << static_cast<std::uint32_t>(std::clamp(255.999 * pixel_color.x(), 0.0, 255.0)) << ' '
		<< static_cast<std::uint32_t>(std::clamp(255.999 * pixel_color.y(), 0.0, 255.0)) << ' '
		<< static_cast<std::uint32_t>(std::clamp(255.999 * pixel_color.z(), 0.0, 255.0)) << '\n';
}

#endif // !COLOR_H
