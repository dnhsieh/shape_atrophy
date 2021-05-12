#include "struct.h"

void reactionTauActivity(double *, double *, double *, int);
void reactionTauActivity(double *, double *, double *, double *, int);

void reactionTauActivity(double *d_actFcnVec, double *d_tauVec, fcndata &fcnObj)
{
	int     lmkNum         = fcnObj.prm.lmkNum;
	double *h_reaActPrmVec = fcnObj.prm.h_reaActPrmVec;

	reactionTauActivity(d_actFcnVec, d_tauVec, h_reaActPrmVec, lmkNum);

	return;
}

void reactionTauActivity(double *d_actFcnVec, double *d_actDotVec, double *d_tauVec, fcndata &fcnObj)
{
	int     lmkNum         = fcnObj.prm.lmkNum;
	double *h_reaActPrmVec = fcnObj.prm.h_reaActPrmVec;

	reactionTauActivity(d_actFcnVec, d_actDotVec, d_tauVec, h_reaActPrmVec, lmkNum);

	return;
}
