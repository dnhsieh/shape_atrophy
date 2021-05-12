#include <cmath>
#include "matvec.h"
#include "struct.h"
#include "constants.h"

__global__ void frameNmlKernel(double *d_nmlMat, double *d_lmkMat, int *d_nmlVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector nmlSumVec = {0.0, 0.0, 0.0};

		for ( int nmlIdx = 0; nmlIdx < NMLNUM; ++nmlIdx )
		{
			int q0Idx = d_nmlVtxMat[(3 * nmlIdx    ) * elmNum + elmIdx];
			int q1Idx = d_nmlVtxMat[(3 * nmlIdx + 1) * elmNum + elmIdx];
			int q2Idx = d_nmlVtxMat[(3 * nmlIdx + 2) * elmNum + elmIdx];

			vector q0Vec, q1Vec, q2Vec;
			getVector(q0Vec, d_lmkMat, q0Idx, lmkNum);
			getVector(q1Vec, d_lmkMat, q1Idx, lmkNum);
			getVector(q2Vec, d_lmkMat, q2Idx, lmkNum);

			vector q10Vec, q20Vec;
			vectorSubtract(q10Vec, q1Vec, q0Vec);
			vectorSubtract(q20Vec, q2Vec, q0Vec);

			vector crsVec;
			crossProduct(crsVec, q10Vec, q20Vec);

			double crsLen = eucnorm(crsVec);
			nmlSumVec.x += crsVec.x / crsLen;
			nmlSumVec.y += crsVec.y / crsLen;
			nmlSumVec.z += crsVec.z / crsLen;
		}

		double nmlLen = eucnorm(nmlSumVec);
		d_nmlMat[             elmIdx] = nmlSumVec.x / nmlLen;
		d_nmlMat[    elmNum + elmIdx] = nmlSumVec.y / nmlLen;
		d_nmlMat[2 * elmNum + elmIdx] = nmlSumVec.z / nmlLen;
	}

	return;
}

__global__ void frameTsvKernel(double *d_tsvMat, double *d_lmkMat, int *d_tsvVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector tsvSumVec = {0.0, 0.0, 0.0};

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
			tsvSumVec.z += difVec.z / difLen;
		}

		double tsvLen = eucnorm(tsvSumVec);
		d_tsvMat[             elmIdx] = tsvSumVec.x / tsvLen;
		d_tsvMat[    elmNum + elmIdx] = tsvSumVec.y / tsvLen;
		d_tsvMat[2 * elmNum + elmIdx] = tsvSumVec.z / tsvLen;
	}

	return;
}

void computeFrame(double *d_nmlMat, double *d_tsvMat, double *d_lmkMat,
                  int *d_nmlVtxMat, int *d_tsvVtxMat, int lmkNum, int elmNum) 
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	frameNmlKernel <<<blkNum, BLKDIM>>> (d_nmlMat, d_lmkMat, d_nmlVtxMat, lmkNum, elmNum);
	frameTsvKernel <<<blkNum, BLKDIM>>> (d_tsvMat, d_lmkMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}

// ---

__global__ void frameNmlKernel(double *d_nmlMat, double *d_nmlLenVec, double *d_lmkMat,
                               int *d_nmlVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector nmlSumVec = {0.0, 0.0, 0.0};

		for ( int nmlIdx = 0; nmlIdx < NMLNUM; ++nmlIdx )
		{
			int q0Idx = d_nmlVtxMat[(3 * nmlIdx    ) * elmNum + elmIdx];
			int q1Idx = d_nmlVtxMat[(3 * nmlIdx + 1) * elmNum + elmIdx];
			int q2Idx = d_nmlVtxMat[(3 * nmlIdx + 2) * elmNum + elmIdx];

			vector q0Vec, q1Vec, q2Vec;
			getVector(q0Vec, d_lmkMat, q0Idx, lmkNum);
			getVector(q1Vec, d_lmkMat, q1Idx, lmkNum);
			getVector(q2Vec, d_lmkMat, q2Idx, lmkNum);

			vector q10Vec, q20Vec;
			vectorSubtract(q10Vec, q1Vec, q0Vec);
			vectorSubtract(q20Vec, q2Vec, q0Vec);

			vector crsVec;
			crossProduct(crsVec, q10Vec, q20Vec);

			double crsLen = eucnorm(crsVec);
			nmlSumVec.x += crsVec.x / crsLen;
			nmlSumVec.y += crsVec.y / crsLen;
			nmlSumVec.z += crsVec.z / crsLen;
		}

		double nmlLen = eucnorm(nmlSumVec);
		d_nmlLenVec[elmIdx] = nmlLen;

		d_nmlMat[             elmIdx] = nmlSumVec.x / nmlLen;
		d_nmlMat[    elmNum + elmIdx] = nmlSumVec.y / nmlLen;
		d_nmlMat[2 * elmNum + elmIdx] = nmlSumVec.z / nmlLen;
	}

	return;
}

__global__ void frameTsvKernel(double *d_tsvMat, double *d_tsvLenVec, double *d_lmkMat,
                               int *d_tsvVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector tsvSumVec = {0.0, 0.0, 0.0};

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
			tsvSumVec.z += difVec.z / difLen;
		}

		double tsvLen = eucnorm(tsvSumVec);
		d_tsvLenVec[elmIdx] = tsvLen;

		d_tsvMat[             elmIdx] = tsvSumVec.x / tsvLen;
		d_tsvMat[    elmNum + elmIdx] = tsvSumVec.y / tsvLen;
		d_tsvMat[2 * elmNum + elmIdx] = tsvSumVec.z / tsvLen;
	}

	return;
}

void computeFrame(double *d_nmlMat, double *d_tsvMat,
                  double *d_nmlLenVec, double *d_tsvLenVec,
                  double *d_lmkMat, int *d_nmlVtxMat, int *d_tsvVtxMat,
                  int lmkNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	frameNmlKernel <<<blkNum, BLKDIM>>> (d_nmlMat, d_nmlLenVec, d_lmkMat, d_nmlVtxMat, lmkNum, elmNum);
	frameTsvKernel <<<blkNum, BLKDIM>>> (d_tsvMat, d_tsvLenVec, d_lmkMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}
