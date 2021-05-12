#include <cublas_v2.h>
#include "struct.h"
#include "constants.h"

void ipcgKernelMultiplication(double *d_KXMat, double *d_XMat, fcndata &fcnObj)
{
	int       lmkNum = fcnObj.prm.lmkNum;
	double *d_knlMat = fcnObj.d_knlMat;

	double oneVal = 1.0;
	double zroVal = 0.0;

	cublasDgemm(fcnObj.blasHdl, CUBLAS_OP_N, CUBLAS_OP_N, lmkNum, DIMNUM, lmkNum,
	            &oneVal, d_knlMat, lmkNum, d_XMat, lmkNum, &zroVal, d_KXMat, lmkNum);

	return;
}
