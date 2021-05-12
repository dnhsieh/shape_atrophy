#include "struct.h"

void yankTauActivity(double *, double *, double *, int);
void yankTauActivity(double *, double *, double *, double *, int);

void yankTauActivity(double *d_actFcnVec, double *d_tauVec, fcndata &fcnObj)
{
	int     lmkNum         = fcnObj.prm.lmkNum;
	double *h_ynkActPrmVec = fcnObj.prm.h_ynkActPrmVec;

	yankTauActivity(d_actFcnVec, d_tauVec, h_ynkActPrmVec, lmkNum);

	return;
}

void yankTauActivity(double *d_actFcnVec, double *d_actDotVec, double *d_tauVec, fcndata &fcnObj)
{
	int     lmkNum         = fcnObj.prm.lmkNum;
	double *h_ynkActPrmVec = fcnObj.prm.h_ynkActPrmVec;

	yankTauActivity(d_actFcnVec, d_actDotVec, d_tauVec, h_ynkActPrmVec, lmkNum);

	return;
}
