#include "struct.h"

void computeExternalYank(double *, double *, double *, double *, double *, double *,
                         int *, int *, int *, int *, int, int, int);

void computeExternalYank(double *d_exYMat, double *d_lmkMat, double *d_lmkEdgMat,
                         double *d_ynkActFcnVec, fcndata &fcnObj)
{
	int    lmkNum = fcnObj.prm.lmkNum;
	int    elmNum = fcnObj.prm.elmNum;
	int bdrElmNum = fcnObj.prm.bdrElmNum;

	double *d_exYElmMat = fcnObj.d_exYElmMat;
	double *d_exYBdrMat = fcnObj.d_exYBdrMat;
	int    *d_elmVtxMat = fcnObj.elm.d_elmVtxMat;
	int    *d_vtxElmMat = fcnObj.elm.d_vtxElmMat;
	int    *d_bdrVtxMat = fcnObj.elm.d_bdrVtxMat;
	int    *d_vtxBdrMat = fcnObj.elm.d_vtxBdrMat;

	computeExternalYank(d_exYMat, d_lmkMat, d_lmkEdgMat, d_ynkActFcnVec,
	                    d_exYElmMat, d_exYBdrMat, d_elmVtxMat, d_vtxElmMat,
	                    d_bdrVtxMat, d_vtxBdrMat, lmkNum, elmNum, bdrElmNum);

	return;
}
