#include "matvec.h"
#include "constants.h"

__device__ double intpow(double bseVal, int expInt)
{
	double outVal = 1;
	double powVal = bseVal;

	do
	{
		int quoInt = expInt >> 1;
		int remInt = expInt - 2 * quoInt;

		if ( remInt == 1 ) outVal *= powVal;
		powVal *= powVal;

		expInt = quoInt;
	}
	while ( expInt != 0 );

	return outVal;
}

__global__ void tauInitialKernel(double *d_tauVec,
                                 double *d_lmkMat, double tauCenXVal, double tauCenYVal, double tauHgtVal,
                                 double tauRadVal, int tauPowInt, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		vector lmkVec;
		getVector(lmkVec, d_lmkMat, lmkIdx, lmkNum);

		double tauRad2 = tauRadVal * tauRadVal;

		double xDif   = lmkVec.x - tauCenXVal;
		double yDif   = lmkVec.y - tauCenYVal;
		double radSqu = (xDif * xDif + yDif * yDif) / tauRad2;

		double    tmpVal = (radSqu - 1.0) * (radSqu - 1.0);
		double preTauVal = (radSqu < 1.0 ? 1.0 - intpow(1.0 - tmpVal, tauPowInt) : 0.0);

		d_tauVec[lmkIdx] = tauHgtVal * preTauVal;
	}

	return;
}

void tauInitial(double *d_tauVec,
                double *d_lmkMat, double *h_tauVarVec, double *h_tauPrmVec, int lmkNum)
{
	double tauCenXVal = h_tauVarVec[0];
	double tauCenYVal = h_tauVarVec[1];
	double  tauHgtVal = h_tauVarVec[2];
	double  tauRadVal = h_tauVarVec[3];
	int     tauPowInt = h_tauPrmVec[0];

	int blkNum = (lmkNum - 1) / BLKDIM + 1;
	tauInitialKernel <<<blkNum, BLKDIM>>> (d_tauVec,
	                                       d_lmkMat, tauCenXVal, tauCenYVal, tauHgtVal,
	                                       tauRadVal, tauPowInt, lmkNum);

	return;
}

// ---

__global__ void tauInitialKernel(double *d_tauVec, double *d_duTauMat,
                                 double *d_lmkMat, double tauCenXVal, double tauCenYVal, double tauHgtVal,
                                 double tauRadVal, int tauPowInt, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		double *d_dcxTauVec = d_duTauMat;
		double *d_dcyTauVec = d_duTauMat +     lmkNum;
		double *d_dhTauVec  = d_duTauMat + 2 * lmkNum;

		vector lmkVec;
		getVector(lmkVec, d_lmkMat, lmkIdx, lmkNum);

		double tauRad2 = tauRadVal * tauRadVal;

		double xDif   = lmkVec.x - tauCenXVal;
		double yDif   = lmkVec.y - tauCenYVal;
		double radSqu = (xDif * xDif + yDif * yDif) / tauRad2;

		double    tmpVal = (radSqu - 1.0) * (radSqu - 1.0);
		double preTauVal = (radSqu < 1.0 ? 1.0 - intpow(1.0 - tmpVal, tauPowInt) : 0.0);
		double   DTauVal = tauHgtVal * tauPowInt * intpow(1.0 - tmpVal, tauPowInt - 1) * 2 * (radSqu - 1.0);

		   d_tauVec[lmkIdx] = tauHgtVal * preTauVal;
		d_dcxTauVec[lmkIdx] = (radSqu < 1.0 ? -2.0 * DTauVal / tauRad2 * xDif : 0.0);
		d_dcyTauVec[lmkIdx] = (radSqu < 1.0 ? -2.0 * DTauVal / tauRad2 * yDif : 0.0);
		 d_dhTauVec[lmkIdx] = preTauVal;
	}

	return;
}

void tauInitial(double *d_tauVec, double *d_duTauMat,
                double *d_lmkMat, double *h_tauVarVec, double *h_tauPrmVec, int lmkNum)
{
	double tauCenXVal = h_tauVarVec[0];
	double tauCenYVal = h_tauVarVec[1];
	double  tauHgtVal = h_tauVarVec[2];
	double  tauRadVal = h_tauVarVec[3];
	int     tauPowInt = h_tauPrmVec[0];

	int blkNum = (lmkNum - 1) / BLKDIM + 1;
	tauInitialKernel <<<blkNum, BLKDIM>>> (d_tauVec, d_duTauMat, d_lmkMat,
	                                       tauCenXVal, tauCenYVal, tauHgtVal,
	                                       tauRadVal, tauPowInt, lmkNum);

	return;
}
