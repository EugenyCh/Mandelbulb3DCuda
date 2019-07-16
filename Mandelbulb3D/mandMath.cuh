#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

using namespace std;

struct cuPolarVector
{
	double r, phi, theta;

	__device__ cuPolarVector(double vr, double vphi, double vtheta) : r(vr), phi(vphi), theta(vtheta) {}

	__device__ cuPolarVector(const cuPolarVector& other)
	{
		r = other.r;
		phi = other.phi;
		theta = other.theta;
	}

	__device__ cuPolarVector pow(double n)
	{
		return cuPolarVector(
			std::pow(r, n),
			phi * n,
			theta * n);
	}
};

struct cuEuclidVector
{
	double p[3];

	__device__ cuEuclidVector(double x, double y, double z)
	{
		p[0] = x;
		p[1] = y;
		p[2] = z;
	}

	__device__ cuEuclidVector(const cuEuclidVector& other)
	{
		p[0] = other.p[0];
		p[1] = other.p[1];
		p[2] = other.p[2];
	}

	__device__ cuPolarVector toPolar()
	{
		double x = p[0], y = p[1], z = p[2];
		double r = sqrt(x * x + y * y + z * z);
		double phi = x != 0 ? atan(y / x) : 0.0;
		double theta = r != 0 ? acos(z / r) : 0.0;
		return cuPolarVector(r, phi, theta);
	}

	__device__ double square(void)
	{
		return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
	}

	__device__ void rotate(double angle, int a, int b)
	{
		if (a < 0 || a > 2 || b < 0 || b > 2 || a == b)
			return;
		double s = (a < b) ? 1.0 : -1.0;
		double pa = p[a], pb = p[b];
		p[a] = pa * cos(angle) - s * pb * sin(angle);
		p[b] = s * pa * sin(angle) + pb * cos(angle);
	}

	__device__ cuEuclidVector pow(double n)
	{
		double x, y, z;
		cuPolarVector pv = toPolar().pow(n);
		x = pv.r * sin(pv.theta) * sin(pv.phi);
		y = pv.r * sin(pv.theta) * cos(pv.phi);
		z = pv.r * cos(pv.theta);
		return cuEuclidVector(x, y, z);
	}

	__device__ cuEuclidVector operator+(const cuEuclidVector& other)
	{
		return cuEuclidVector(
			p[0] + other.p[0],
			p[1] + other.p[1],
			p[2] + other.p[2]);
	}
};
