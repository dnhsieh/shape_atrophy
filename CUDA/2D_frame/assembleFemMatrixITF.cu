#include "struct.h"

void assembleFemMatrix(double *, double *, double *, double *, double *,
                       double, double, double *, double *, int *, int *, int, int, int);

void assembleFemMatrix(double *d_ppdMat, double *d_ggdMat,
                       double *d_lmkEdgMat, double *d_tanMat, double *d_tsvMat, fcndata &fcnObj)
{
	int    lmkNum    = fcnObj.prm.lmkNum;
	int    nzrNum    = fcnObj.prm.nzrNum;
	int    elmNum    = fcnObj.prm.elmNum;
	double spdTanVal = fcnObj.prm.spdTanVal;
	double spdTsvVal = fcnObj.prm.spdTsvVal;

	int    *d_femVtxMat = fcnObj.elm.d_femVtxMat;
	int    *d_femIfoMat = fcnObj.elm.d_femIfoMat;
	double *d_ppdElmVec = fcnObj.d_ppdElmVec;
	double *d_ggdElmMat = fcnObj.d_ggdElmMat;

	assembleFemMatrix(d_ppdMat, d_ggdMat, d_lmkEdgMat, d_tanMat, d_tsvMat,
	                  spdTanVal, spdTsvVal, d_ppdElmVec, d_ggdElmMat,
	                  d_femVtxMat, d_femIfoMat, lmkNum, nzrNum, elmNum);

	return;
}
