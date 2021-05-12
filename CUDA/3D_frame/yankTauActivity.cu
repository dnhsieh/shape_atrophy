#include "activity.h"
#include "constants.h"

__global__ void yankActivityKernel(double *d_actFcnVec, double *d_tauVec,
                                   double tauMin, double tauMax, double actMax, double dWidth, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		double tauVal = d_tauVec[lmkIdx];

		double actFcnVal;
		yankActivityFunction(actFcnVal, tauVal, tauMin, tauMax, actMax, dWidth);

		d_actFcnVec[lmkIdx] = actFcnVal;
	}

	return;
}

void yankTauActivity(double *d_actFcnVec, double *d_tauVec, double *h_actPrmVec, int lmkNum)
{
	double tauMin = h_actPrmVec[0];
	double tauMax = h_actPrmVec[1];
	double actMax = h_actPrmVec[2];
	double dWidth = h_actPrmVec[3];

	int blkNum = (lmkNum - 1) / BLKDIM + 1;
	yankActivityKernel <<<blkNum, BLKDIM>>> (d_actFcnVec, d_tauVec,
	                                         tauMin, tauMax, actMax, dWidth, lmkNum);

	return;
}

// - - -

__global__ void yankActivityKernel(double *d_actFcnVec, double *d_actDotVec, double *d_tauVec,
                                   double tauMin, double tauMax, double actMax, double dWidth, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		double tauVal = d_tauVec[lmkIdx];

		double actFcnVal, actDotVal;
		yankActivityFunction(actFcnVal, actDotVal, tauVal, tauMin, tauMax, actMax, dWidth);

		d_actFcnVec[lmkIdx] = actFcnVal;
		d_actDotVec[lmkIdx] = actDotVal;
	}

	return;
}

void yankTauActivity(double *d_actFcnVec, double *d_actDotVec, double *d_tauVec,
                     double *h_actPrmVec, int lmkNum)
{
	double tauMin = h_actPrmVec[0];
	double tauMax = h_actPrmVec[1];
	double actMax = h_actPrmVec[2];
	double dWidth = h_actPrmVec[3];

	int blkNum = (lmkNum - 1) / BLKDIM + 1;
	yankActivityKernel <<<blkNum, BLKDIM>>> (d_actFcnVec, d_actDotVec, d_tauVec,
	                                         tauMin, tauMax, actMax, dWidth, lmkNum);

	return;
}
