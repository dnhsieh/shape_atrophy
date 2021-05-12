#include "struct.h"

void tauInitial(double *, double *, double *, double *, int);
void tauInitial(double *, double *, double *, double *, double *, int);

void tauInitial(double *d_tauVec, double *h_tauVarVec, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;

	double *h_tauPrmVec = fcnObj.prm.h_tauPrmVec;
	double *d_lmkIniMat = fcnObj.prm.d_lmkIniMat;

	tauInitial(d_tauVec, d_lmkIniMat, h_tauVarVec, h_tauPrmVec, lmkNum);

	return;
}

void tauInitial(double *d_tauVec, double *d_duTauMat, double *h_tauVarVec, fcndata &fcnObj)
{
	int lmkNum = fcnObj.prm.lmkNum;

	double *h_tauPrmVec = fcnObj.prm.h_tauPrmVec;
	double *d_lmkIniMat = fcnObj.prm.d_lmkIniMat;

	tauInitial(d_tauVec, d_duTauMat, d_lmkIniMat, h_tauVarVec, h_tauPrmVec, lmkNum);

	return;
}
