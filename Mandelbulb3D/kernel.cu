#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg/CImg.h"
#include "mandMath.cuh"
#include <iostream>
#include <ctime>
#include <iomanip>
#include <string>
#include <sstream>
#include <cmath>

#define PI 3.14159265358979323846

using namespace std;
using namespace cimg_library;

__global__ void kernel(
	unsigned char* buffer,
	int offsetX,
	int offsetY,
	int offsetZ,
	int rectSize,
	int side,
	int iters,
	double power,
	double angleXZ,
	double angleYZ)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= rectSize)
		return;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (y >= rectSize)
		return;
	x += offsetX;
	if (x >= side)
		return;
	y += offsetY;
	if (y >= side)
		return;
	int offset = x + y * side;
	if (buffer[offset * 3] > 0)
		return;

	// Compute a point at this position
	int side1_2 = side >> 1;
	double bailout = pow(2.0, 1.0 / (power - 1.0));
	double bailout2 = bailout * bailout;
	double cx = (bailout * (x - side1_2)) / side1_2;
	double cy = (bailout * (y - side1_2)) / side1_2;

	// Iterating
	bool belongs = false;
	double sqr;
	int z1 = side - 1 - offsetZ;
	int z2 = max(side1_2 - 1, side - 1 - rectSize - offsetZ);
	for (int z = z1; z >= z2; --z)
	{
		double cz = (bailout * (z - side1_2)) / side1_2;
		cuEuclidVector cVec(cx, cy, cz);
		cVec.rotate(angleXZ, 0, 2);
		cVec.rotate(angleYZ, 1, 2);
		sqr = cVec.square();
		cuEuclidVector vec(cVec);
		for (int i = 0; i < iters; ++i)
			vec = vec.pow(power) + cVec;
		if (vec.square() <= bailout2)
		{
			belongs = true;
			break;
		}
	}

	// Setting the point color
	if (belongs)
	{
		double k = sqr / bailout2;
		buffer[offset * 3] = (unsigned char)(1 + k * 127);
		buffer[offset * 3 + 1] = (unsigned char)(k * 127);
		buffer[offset * 3 + 2] = (unsigned char)((2.0 * k - k * k) * 255);
	}
}

int main(int argc, char** argv)
{
	// Settting
	if (argc != 4 && argc != 6)
	{
		cout << "Usage: MandBulb <side width in pixels> <power> <iterations> [<angleXZ> <angleYZ>]" << endl;
		return 1;
	}

	int side;
	int power;
	int iters;
	double angleXZ = 0.0;
	double angleYZ = 0.0;

	try
	{
		side = stoi(argv[1]);
		if (side < 32)
		{
			cout << "Width must be integer >= 32" << endl;
			return 1;
		}
	}
	catch (...)
	{
		cout << "Width must integer" << endl;
		return 1;
	}

	try
	{
		power = stoi(argv[2]);
		if (power < 2)
		{
			cout << "Power must be integer >= 2" << endl;
			return 1;
		}
	}
	catch (...)
	{
		cout << "Power must be integer" << endl;
		return 1;
	}

	try
	{
		iters = stoi(argv[3]);
		if (iters < 0)
		{
			cout << "Iterations must be integer >= 0" << endl;
			return 1;
		}
	}
	catch (...)
	{
		cout << "Iterations must be integer" << endl;
		return 1;
	}

	if (argc == 6)
	{
		try
		{
			angleXZ = stod(argv[4]);
		}
		catch (...)
		{
			cout << "AngleXZ must be floating-point number" << endl;
			return 1;
		}

		try
		{
			angleYZ = stod(argv[5]);
		}
		catch (...)
		{
			cout << "AngleYZ must be floating-point number" << endl;
			return 1;
		}
	}

	angleXZ = angleXZ * PI / 180.0;
	angleYZ = angleYZ * PI / 180.0;

	// Initializing
	const int sz = side * side;
	unsigned char* buffer = new unsigned char[sz * 3];
	unsigned char* dev_buffer;
	CImg<unsigned char> image(side, side, 1, 3, 0);

	cudaError status;
	if ((status = cudaMalloc((void**)& dev_buffer, sz * 3)) != cudaSuccess)
	{
		cerr << "Error on creating buffer of pixels in GPU" << endl;
		return status;
	}

	// Rendering
	time_t tStart = time(0);
	dim3 blocks(5, 5);
	dim3 threads(32, 32);
	int gridDim = (side + 159) / 160;
	int index = 0;
	for (int yi = 0; yi < gridDim; ++yi)
		for (int xi = 0; xi < gridDim; ++xi)
			for (int zi = 0; zi < gridDim; ++zi)
			{
				cout << "\rRendering " << (++index) << " / " << (gridDim * gridDim * gridDim);
				kernel<<<blocks, threads>>> (
					dev_buffer,
					xi * 160,
					yi * 160,
					zi * 160,
					160,
					side,
					iters,
					power,
					angleXZ,
					angleYZ);
				cudaDeviceSynchronize();
			}

	time_t tFinish = time(0);
	double tDelta = difftime(tFinish, tStart);
	cout << "\nIt tooks " << setprecision(3) << showpoint << tDelta << " seconds" << endl;

	// Copying device buffer to host
	cout << "Moving" << endl;
	if ((status = cudaMemcpy((void*)buffer, dev_buffer, sz * 3, cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		cerr << "Error on getting buffer of pixels from GPU: " << status << endl;
		return status;
	}

	// Freeing
	cudaFree(dev_buffer);

	// Filling
	cout << "Filling" << endl;
	for (int y = 0; y < side; ++y)
		for (int x = 0; x < side; ++x)
			image.draw_point(x, y, &buffer[3 * (y * side + x)]);

	// Saving
	stringstream filename;
	filename << "Mandelbulb-"
		<< side << "x" << side
		<< "-n" << power
		<< "-i" << iters
		<< "-xz" << (int)(angleXZ * 180.0 / PI)
		<< "-yz" << (int)(angleYZ * 180.0 / PI)
		<< ".bmp";
	cout << "Saving to " << filename.str() << endl;
	image.save_bmp(filename.str().c_str());

	return 0;
}
