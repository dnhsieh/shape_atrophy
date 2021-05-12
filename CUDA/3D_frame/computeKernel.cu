#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void gaussian(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);
		double dstSqu = eucdistSqu(qiVec, qjVec);

		dstSqu /= knlWidth * knlWidth;
		d_knlMat[colIdx * lmkNum + rowIdx] = exp(-dstSqu);
	}

	return;
}

__global__ void matern0(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);
		double dstVal = eucdist(qiVec, qjVec);

		dstVal /= knlWidth;
		d_knlMat[colIdx * lmkNum + rowIdx] = exp(-dstVal);
	}

	return;
}

__global__ void matern1(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);
		double dstVal = eucdist(qiVec, qjVec);

		dstVal /= knlWidth;
		d_knlMat[colIdx * lmkNum + rowIdx] = (1.0 + dstVal) * exp(-dstVal);
	}

	return;
}

__global__ void matern2(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);
		double dstVal = eucdist(qiVec, qjVec);

		dstVal /= knlWidth;
		d_knlMat[colIdx * lmkNum + rowIdx] = (3.0 + dstVal * (3.0 + dstVal)) / 3.0 * exp(-dstVal);
	}

	return;
}

__global__ void matern3(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);
		double dstVal = eucdist(qiVec, qjVec);

		dstVal /= knlWidth;
		d_knlMat[colIdx * lmkNum + rowIdx] = 
		   (15.0 + dstVal * (15.0 + dstVal * (6.0 + dstVal))) / 15.0 * exp(-dstVal);
	}

	return;
}

__global__ void matern4(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);
		double dstVal = eucdist(qiVec, qjVec);

		dstVal /= knlWidth;
		d_knlMat[colIdx * lmkNum + rowIdx] = 
		   (105.0 + dstVal * (105.0 + dstVal * (45.0 + dstVal * (10.0 + dstVal)))) / 105.0 * exp(-dstVal);
	}

	return;
}

void computeKernel(double *d_knlMat, double *d_lmkMat, int knlOrder, double knlWidth, int lmkNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	int  gridRow = (lmkNum - 1) / BLKROW + 1;
	dim3 blkNum(gridRow, gridRow);
	dim3 blkDim( BLKROW,  BLKROW);

	switch ( knlOrder )
	{
		case -1:
			gaussian <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 0:
			matern0 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 1:
			matern1 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 2:
			matern2 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 3:
			matern3 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 4:
			matern4 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;
	}

	return;
}

