#include "struct.h"

void applyFrameElasticOperator(double *, double *, double *, double *, double *, double *,
                               double *, double *, int *, int *, int, int);

void applyFrameElasticOperator(double *d_elYMat, double *d_vlcMat, double *d_lmkEdgMat,
                               double *d_tanMat, double *d_tsvMat, fcndata &fcnObj)
{
	int       lmkNum = fcnObj.prm.lmkNum;
	int       elmNum = fcnObj.prm.elmNum;
	double *h_modVec = fcnObj.prm.h_modVec;

	int    *d_elmVtxMat = fcnObj.elm.d_elmVtxMat;
	int    *d_vtxElmMat = fcnObj.elm.d_vtxElmMat;
	double *d_epvMat    = fcnObj.d_epvMat;
	double *d_elYElmMat = fcnObj.d_elYElmMat;

	applyFrameElasticOperator(d_elYMat, d_lmkEdgMat, d_tanMat, d_tsvMat, h_modVec,
	                          d_vlcMat, d_epvMat, d_elYElmMat,
	                          d_elmVtxMat, d_vtxElmMat, lmkNum, elmNum);

	return;
}
