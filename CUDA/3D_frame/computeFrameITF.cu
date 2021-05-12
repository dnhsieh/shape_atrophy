#include "struct.h"

void computeFrame(double *, double *, double *, int *, int *, int, int);
void computeFrame(double *, double *, double *, double *, double *, int *, int *, int, int);

void computeFrame(double *d_nmlMat, double *d_tsvMat, double *d_lmkMat, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;
	int elmNum = fcnObj.prm.elmNum;

	int *d_nmlVtxMat = fcnObj.elm.d_nmlVtxMat;
	int *d_tsvVtxMat = fcnObj.elm.d_tsvVtxMat;

	computeFrame(d_nmlMat, d_tsvMat, d_lmkMat, d_nmlVtxMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}

void computeFrame(double *d_nmlMat, double *d_tsvMat, double *d_nmlLenVec, double *d_tsvLenVec,
                  double *d_lmkMat, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;
	int elmNum = fcnObj.prm.elmNum;

	int *d_nmlVtxMat = fcnObj.elm.d_nmlVtxMat;
	int *d_tsvVtxMat = fcnObj.elm.d_tsvVtxMat;

	computeFrame(d_nmlMat, d_tsvMat, d_nmlLenVec, d_tsvLenVec,
	             d_lmkMat, d_nmlVtxMat, d_tsvVtxMat, lmkNum, elmNum);

	return;
}
