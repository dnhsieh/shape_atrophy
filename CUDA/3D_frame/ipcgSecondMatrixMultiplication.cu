#include "constants.h"
#include "struct.h"

void applyFrameElasticOperator(double *, double *, double *, double *, double *, fcndata &); 
void applyPenaltyOperator(double *, double *, double *, fcndata &);
void vectoraxpby(double *, double, double *, double, double *, int);

void ipcgSecondMatrixMultiplication(double *d_AxVec, double *d_xVec, fcndata &fcnObj)
{
	int    lmkNum = fcnObj.prm.lmkNum;
	double btmWgt = fcnObj.prm.btmWgt;

	double *d_elYMat = fcnObj.d_elYMat;
	double *d_pnlMat = fcnObj.d_pnlMat;

	applyFrameElasticOperator(d_elYMat, d_xVec, fcnObj.d_lmkNowEdgMat,
	                          fcnObj.d_nmlNowMat, fcnObj.d_tsvNowMat, fcnObj);

	applyPenaltyOperator(d_pnlMat, fcnObj.d_lmkNowBtmMat, d_xVec, fcnObj); 

	vectoraxpby(d_AxVec, 1.0, d_elYMat, btmWgt, d_pnlMat, lmkNum * DIMNUM);

	return;
}
