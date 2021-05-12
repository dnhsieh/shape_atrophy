#include <cusolverDn.h>
#include "constants.h"
#include "struct.h"

void applyFrameElasticOperator(double *, double *, double *, double *, double *, fcndata &); 
void applyPenaltyOperator(double *, double *, double *, fcndata &);

__global__ void ipcgMatSumKernel(double *d_AxVec, double ldmWgt, double *d_KivMat,
                                 double *d_elYMat, double btmWgt, double *d_pnlMat, int ttlNum)
{
	int ttlIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( ttlIdx < ttlNum )
	{
		d_AxVec[ttlIdx] =  ldmWgt * d_KivMat[ttlIdx]
		                 + d_elYMat[ttlIdx] + btmWgt * d_pnlMat[ttlIdx];
	}

	return;
}

void ipcgWholeMatrixMultiplication(double *d_AxVec, double *d_xVec, fcndata &fcnObj)
{
	int    lmkNum = fcnObj.prm.lmkNum;
	double ldmWgt = fcnObj.prm.ldmWgt;
	double btmWgt = fcnObj.prm.btmWgt;

	double *d_KivMat = fcnObj.d_KivMat;
	double *d_elYMat = fcnObj.d_elYMat;
	double *d_pnlMat = fcnObj.d_pnlMat;

	cudaMemcpy(d_KivMat, d_xVec, sizeof(double) * lmkNum * DIMNUM, cudaMemcpyDeviceToDevice);
	cusolverDnDpotrs(fcnObj.solvHdl, CUBLAS_FILL_MODE_LOWER, lmkNum, DIMNUM, fcnObj.d_knLMat, lmkNum,
	                 d_KivMat, lmkNum, fcnObj.d_status);

	applyFrameElasticOperator(d_elYMat, d_xVec, fcnObj.d_lmkNowEdgMat,
	                          fcnObj.d_tanNowMat, fcnObj.d_tsvNowMat, fcnObj);

	applyPenaltyOperator(d_pnlMat, fcnObj.d_lmkNowBtmMat, d_xVec, fcnObj); 

	int blkNum = (lmkNum * DIMNUM - 1) / BLKDIM + 1;
	ipcgMatSumKernel <<<blkNum, BLKDIM>>> (d_AxVec, ldmWgt, d_KivMat,
	                                       d_elYMat, btmWgt, d_pnlMat, lmkNum * DIMNUM);

	return;
}
