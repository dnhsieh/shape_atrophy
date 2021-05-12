#include "struct.h"

void assembleFemVector(double *, double *, double *, double *, int *, int *, int, int);

void assembleFemVector(double *d_rpdVec, double *d_lmkEdgMat, double *d_actFcnVec, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;
	int elmNum = fcnObj.prm.elmNum;

	int    *d_elmVtxMat = fcnObj.elm.d_elmVtxMat;
	int    *d_vtxElmMat = fcnObj.elm.d_vtxElmMat;
	double *d_rpdElmMat = fcnObj.d_rpdElmMat;

	assembleFemVector(d_rpdVec, d_lmkEdgMat, d_actFcnVec,
	                  d_rpdElmMat, d_elmVtxMat, d_vtxElmMat, lmkNum, elmNum);

	return;
}
