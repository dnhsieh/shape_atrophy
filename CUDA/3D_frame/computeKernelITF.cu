#include "struct.h"

void computeKernel(double *, double *, int, double, int);

void computeKernel(double *d_knlMat, double *d_lmkMat, fcndata &fcnObj)
{
	int    knlOrder = fcnObj.prm.knlOrder;
	double knlWidth = fcnObj.prm.knlWidth;
	int    lmkNum   = fcnObj.prm.lmkNum;

	computeKernel(d_knlMat, d_lmkMat, knlOrder, knlWidth, lmkNum);

	return;
}
