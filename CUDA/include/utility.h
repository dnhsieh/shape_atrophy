#ifndef UTILITY_H
#define UTILITY_H

#include <cmath>
#include "matvec.h"

void dsum(double *, double *, double *, int);

#ifdef DIM2

inline __device__ double computeVolume(vector q10Vec, vector q20Vec)
{
	double detVal = det(q10Vec, q20Vec);
	double volVal = 0.5 * fabs(detVal);

	return volVal;
}

#elif DIM3

inline __device__ double computeVolume(vector q10Vec, vector q20Vec, vector q30Vec)
{
	double detVal = det(q10Vec, q20Vec, q30Vec);
	double volVal = fabs(detVal) / 6.0;

	return volVal;
}

#endif

#endif
