#include "matvec.h"
#include "constants.h"

__global__ void frameTanKernel(double *d_tanMat, double *d_lmkMat,
                               int *d_tanVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector tanSumVec = {0.0, 0.0};

		for ( int tanIdx = 0; tanIdx < TANNUM; ++tanIdx )
		{
			int lftIdx = d_tanVtxMat[(2 * tanIdx    ) * elmNum + elmIdx];
			int rgtIdx = d_tanVtxMat[(2 * tanIdx + 1) * elmNum + elmIdx];

			vector lftVec, rgtVec;
			getVector(lftVec, d_lmkMat, lftIdx, lmkNum);
			getVector(rgtVec, d_lmkMat, rgtIdx, lmkNum);

			vector difVec;
			vectorSubtract(difVec, rgtVec, lftVec);

			double difLen = eucnorm(difVec);
			tanSumVec.x += difVec.x / difLen;
			tanSumVec.y += difVec.y / difLen;
		}

		double tanLen = eucnorm(tanSumVec);
		d_tanMat[         elmIdx] = tanSumVec.x / tanLen;
		d_tanMat[elmNum + elmIdx] = tanSumVec.y / tanLen;
	}

	return;
}

__global__ void frameTsvKernel(double *d_tsvMat, double *d_lmkMat,
                               int *d_tsvVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector tsvSumVec = {0.0, 0.0};

		for ( int tsvIdx = 0; tsvIdx < TSVNUM; ++tsvIdx )
		{
			int dwnIdx = d_tsvVtxMat[(2 * tsvIdx    ) * elmNum + elmIdx];
			int uppIdx = d_tsvVtxMat[(2 * tsvIdx + 1) * elmNum + elmIdx];

			vector dwnVec, uppVec;
			getVector(dwnVec, d_lmkMat, dwnIdx, lmkNum);
			getVector(uppVec, d_lmkMat, uppIdx, lmkNum);

			vector difVec;
			vectorSubtract(difVec, uppVec, dwnVec);

			double difLen = eucnorm(difVec);
			tsvSumVec.x += difVec.x / difLen;
			tsvSumVec.y += difVec.y / difLen;
		}

		double tsvLen = eucnorm(tsvSumVec);
		d_tsvMat[         elmIdx] = tsvSumVec.x / tsvLen;
		d_tsvMat[elmNum + elmIdx] = tsvSumVec.y / tsvLen;
	}

	return;
}

void computeFrame(double *d_tanMat, double *d_tsvMat,
                  double *d_lmkMat, int *d_tanVtxMat, int *d_tsvVtxMat, int lmkNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	frameTanKernel <<<blkNum, BLKDIM>>> (d_tanMat, d_lmkMat, d_tanVtxMat, lmkNum, elmNum);
	frameTsvKernel <<<blkNum, BLKDIM>>> (d_tsvMat, d_lmkMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}

// - - -

__global__ void frameTanKernel(double *d_tanMat, double *d_tanLenVec, double *d_lmkMat,
                               int *d_tanVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector tanSumVec = {0.0, 0.0};
		for ( int tanIdx = 0; tanIdx < TANNUM; ++tanIdx )
		{
			int lftIdx = d_tanVtxMat[(2 * tanIdx    ) * elmNum + elmIdx];
			int rgtIdx = d_tanVtxMat[(2 * tanIdx + 1) * elmNum + elmIdx];

			vector lftVec, rgtVec;
			getVector(lftVec, d_lmkMat, lftIdx, lmkNum);
			getVector(rgtVec, d_lmkMat, rgtIdx, lmkNum);

			vector difVec;
			vectorSubtract(difVec, rgtVec, lftVec);

			double difLen = eucnorm(difVec);
			tanSumVec.x += difVec.x / difLen;
			tanSumVec.y += difVec.y / difLen;
		}

		double tanLen = eucnorm(tanSumVec);
		d_tanLenVec[elmIdx] = tanLen;

		d_tanMat[         elmIdx] = tanSumVec.x / tanLen;
		d_tanMat[elmNum + elmIdx] = tanSumVec.y / tanLen;
	}

	return;
}

__global__ void frameTsvKernel(double *d_tsvMat, double *d_tsvLenVec, double *d_lmkMat,
                               int *d_tsvVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector tsvSumVec = {0.0, 0.0};
		for ( int tsvIdx = 0; tsvIdx < TSVNUM; ++tsvIdx )
		{
			int dwnIdx = d_tsvVtxMat[(2 * tsvIdx    ) * elmNum + elmIdx];
			int uppIdx = d_tsvVtxMat[(2 * tsvIdx + 1) * elmNum + elmIdx];

			vector dwnVec, uppVec;
			getVector(dwnVec, d_lmkMat, dwnIdx, lmkNum);
			getVector(uppVec, d_lmkMat, uppIdx, lmkNum);

			vector difVec;
			vectorSubtract(difVec, uppVec, dwnVec);

			double difLen = eucnorm(difVec);
			tsvSumVec.x += difVec.x / difLen;
			tsvSumVec.y += difVec.y / difLen;
		}

		double tsvLen = eucnorm(tsvSumVec);
		d_tsvLenVec[elmIdx] = tsvLen;

		d_tsvMat[         elmIdx] = tsvSumVec.x / tsvLen;
		d_tsvMat[elmNum + elmIdx] = tsvSumVec.y / tsvLen;
	}

	return;
}

void computeFrame(double *d_tanMat, double *d_tsvMat, double *d_tanLenVec, double *d_tsvLenVec,
                  double *d_lmkMat, int *d_tanVtxMat, int *d_tsvVtxMat, int lmkNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	frameTanKernel <<<blkNum, BLKDIM>>> (d_tanMat, d_tanLenVec, d_lmkMat, d_tanVtxMat, lmkNum, elmNum);
	frameTsvKernel <<<blkNum, BLKDIM>>> (d_tsvMat, d_tsvLenVec, d_lmkMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}
