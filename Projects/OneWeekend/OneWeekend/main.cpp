
#pragma region
//includes
#include <iostream>
#pragma endregion

//indicate the image resolution
const int width = 256;
const int height = 256;

int main()
{
	//render into the image .ppm format
	std::cout << "P3\n" << width << ' ' << height << "\n255\n";
	//wirte the image from left to right, top to bottom (first row to the last row)
	for (int i = height - 1; i >= 0; --i)
	{
		for (int j = 0; j < width; ++j)
		{
			auto r = double(j) / (width - 1);
			auto g = double(i) / (height - 1);
			double b = 0.25;
			int ir = static_cast<int> (255.999 * r);
			int ig = static_cast<int> (255.999 * g);   //make sure it reach to 255
			int ib = static_cast<int> (255.999 * b);

			//simple ppm format, tone map to 0-255
			std::cout << ir << ' ' << ig << ' ' << ib << '\n';
		}
	}
	return 0;
}
