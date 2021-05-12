#include "constants.h"

__global__ void axpbyKernel(double *d_out, double aVal, double *d_x, double bVal, double *d_y, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < len )
		d_out[idx] = aVal * d_x[idx] + bVal * d_y[idx];

	return;
}

void vectoraxpby(double *d_out, double aVal, double *d_x, double bVal, double *d_y, int len)
{
	int blkNum = (len - 1) / BLKDIM + 1;
	axpbyKernel <<<blkNum, BLKDIM>>> (d_out, aVal, d_x, bVal, d_y, len);

	return;
}
