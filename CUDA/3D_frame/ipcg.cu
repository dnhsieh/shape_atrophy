#include <cstdio>
#include <cublas_v2.h>
#include "struct.h"
#include "constants.h"

void ipcgWholeMatrixMultiplication(double *, double *, fcndata &);
void ipcgSecondMatrixMultiplication(double *, double *, fcndata &);
void ipcgKernelMultiplication(double *, double *, fcndata &);
void vectoraxpby(double *, double, double *, double, double *, int);

int ipcg(double *d_xVec, double *d_bVec, fcndata &fcnObj)
{
	int    varNum = fcnObj.pcg.varNum;
	int    itrMax = fcnObj.pcg.itrMax;
	double tolSqu = fcnObj.pcg.tolSqu;

	double *d_AdVec  = fcnObj.pcg.d_AdVec;
	double *d_BdVec  = fcnObj.pcg.d_BdVec;
	double *d_KidVec = fcnObj.pcg.d_KidVec;
	double *d_rVec   = fcnObj.pcg.d_rVec;
	double *d_KrVec  = fcnObj.pcg.d_KrVec;
	double *d_dVec   = fcnObj.pcg.d_dVec;

	double h_bSqu;
	cublasDdot(fcnObj.blasHdl, varNum, d_bVec, 1, d_bVec, 1, &h_bSqu);
	if ( h_bSqu == 0.0 )
	{
		cudaMemset(d_xVec, 0, sizeof(double) * varNum);	
		return 0;
	}

	double relTolSqu = h_bSqu * tolSqu;

	ipcgWholeMatrixMultiplication(d_AdVec, d_xVec, fcnObj);
	vectoraxpby(d_rVec, 1.0, d_bVec, -1.0, d_AdVec, varNum);
	
	double h_rSquNow;
	cublasDdot(fcnObj.blasHdl, varNum, d_rVec, 1, d_rVec, 1, &h_rSquNow);
	if ( h_rSquNow < relTolSqu ) return 0;

	cudaMemcpy(d_KidVec, d_rVec, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
	ipcgKernelMultiplication(d_KrVec, d_rVec, fcnObj);
	cudaMemcpy(d_dVec, d_KrVec, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);

	double h_rKrNow;
	cublasDdot(fcnObj.blasHdl, varNum, d_rVec, 1, d_KrVec, 1, &h_rKrNow);

	if ( h_rKrNow == 0.0 )
	{
		printf("ipcg stopped at relative residual %e because r^T K r is zero. K is singular.\n",
		       sqrt(h_rSquNow / h_bSqu));
		return 1;
	}

	for ( int itrIdx = 0; itrIdx < itrMax; ++itrIdx )
	{
		ipcgSecondMatrixMultiplication(d_BdVec, d_dVec, fcnObj);
		vectoraxpby(d_AdVec, fcnObj.prm.ldmWgt, d_KidVec, 1.0, d_BdVec, varNum);

		double h_dAdVal;
		cublasDdot(fcnObj.blasHdl, varNum, d_dVec, 1, d_AdVec, 1, &h_dAdVal);

		if ( h_dAdVal == 0.0 )
		{
			printf("ipcg stopped at relative residual %e because d^T A d is zero. A is singular.\n",
			       sqrt(h_rSquNow / h_bSqu));
			return 1;
		}

		double xStpLen = h_rKrNow / h_dAdVal;
		vectoraxpby(d_xVec, 1.0, d_xVec,  xStpLen, d_dVec,  varNum);
		vectoraxpby(d_rVec, 1.0, d_rVec, -xStpLen, d_AdVec, varNum);

		double h_rSquNxt;
		cublasDdot(fcnObj.blasHdl, varNum, d_rVec, 1, d_rVec, 1, &h_rSquNxt);
		if ( h_rSquNxt < relTolSqu ) return 0;

		ipcgKernelMultiplication(d_KrVec, d_rVec, fcnObj);

		double h_rKrNxt;
		cublasDdot(fcnObj.blasHdl, varNum, d_rVec, 1, d_KrVec, 1, &h_rKrNxt);

		if ( h_rKrNxt == 0.0 )
		{
			printf("ipcg stopped at relative residual %e because r^T K r is zero. K is singular.\n",
			       sqrt(h_rSquNxt / h_bSqu));
			return 1;
		}

		double dStpLen = h_rKrNxt / h_rKrNow;
		vectoraxpby(d_KidVec, 1.0, d_rVec,  dStpLen, d_KidVec, varNum);
		vectoraxpby(d_dVec,   1.0, d_KrVec, dStpLen, d_dVec,   varNum);
	
		h_rSquNow = h_rSquNxt;
		h_rKrNow    = h_rKrNxt;
	}

	printf("ipcg did not converge, stopped at relative residual %e.\n", sqrt(h_rSquNow / h_bSqu));
	return 1;
}
