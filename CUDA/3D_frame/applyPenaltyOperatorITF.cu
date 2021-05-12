#include "struct.h"

void applyPenaltyOperator(double *, double *, double *, double *,
                          int *, int *, int, int, int);

void applyPenaltyOperator(double *d_pnlMat, double *d_lmkBtmMat, double *d_vlcMat, fcndata &fcnObj)
{
	int    lmkNum = fcnObj.prm.lmkNum;
	int btmElmNum = fcnObj.prm.btmElmNum;
	int btmLmkNum = fcnObj.prm.btmLmkNum;

	double *d_pnlBtmMat = fcnObj.d_pnlBtmMat;
	int    *d_btmVtxMat = fcnObj.elm.d_btmVtxMat;
	int    *d_vtxBtmMat = fcnObj.elm.d_vtxBtmMat;

	applyPenaltyOperator(d_pnlMat, d_lmkBtmMat, d_vlcMat,
	                     d_pnlBtmMat, d_btmVtxMat, d_vtxBtmMat,
	                     lmkNum, btmElmNum, btmLmkNum);

	return;
}
