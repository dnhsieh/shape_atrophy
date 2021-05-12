#include "struct.h"

void computeFrame(double *, double *, double *, int *, int *, int, int);
void computeFrame(double *, double *, double *, double *, double *, int *, int *, int, int);

void computeFrame(double *d_tanMat, double *d_tsvMat, double *d_lmkMat, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;
	int elmNum = fcnObj.prm.elmNum;

	int *d_tanVtxMat = fcnObj.elm.d_tanVtxMat;
	int *d_tsvVtxMat = fcnObj.elm.d_tsvVtxMat;

	computeFrame(d_tanMat, d_tsvMat, d_lmkMat, d_tanVtxMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}

void computeFrame(double *d_tanMat, double *d_tsvMat, double *d_tanLenVec, double *d_tsvLenVec,
                  double *d_lmkMat, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;
	int elmNum = fcnObj.prm.elmNum;

	int *d_tanVtxMat = fcnObj.elm.d_tanVtxMat;
	int *d_tsvVtxMat = fcnObj.elm.d_tsvVtxMat;

	computeFrame(d_tanMat, d_tsvMat, d_tanLenVec, d_tsvLenVec,
	             d_lmkMat, d_tanVtxMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}
