#include "struct.h"

void varifold(double *, double *, int *, double *, double *, double *, char, double, char, double,
              double *, double *, double *, double *, double *, int, int, int);

//void varifold(double *, double *, double *, int *, int *, double *, double *, double *,
//              char, double, char, double, double *, double *, double *,
//              double *, double *, double *, double *, int, int, int);

void varifold(double *h_vfdPtr, double *d_dfmLmkPosMat, fcndata &fcnObj)
{
	int dfmLmkNum = fcnObj.prm.vfdLmkNum;
	int dfmElmNum = fcnObj.prm.vfdElmNum;
	int tgtElmNum = fcnObj.tgt.vfdElmNum;

	char   cenKnlType  = fcnObj.vfd.cenKnlType;
	double cenKnlWidth = fcnObj.vfd.cenKnlWidth;
	char   dirKnlType  = fcnObj.vfd.dirKnlType;
	double dirKnlWidth = fcnObj.vfd.dirKnlWidth;

	int    *d_dfmElmVtxMat = fcnObj.elm.d_vfdVtxMat;
	double *d_tgtCenPosMat = fcnObj.tgt.d_cenPosMat;
	double *d_tgtUniDirMat = fcnObj.tgt.d_uniDirMat;
	double *d_tgtElmVolVec = fcnObj.tgt.d_elmVolVec;
	double *d_dfmCenPosMat = fcnObj.d_dfmCenPosMat;
	double *d_dfmUniDirMat = fcnObj.d_dfmUniDirMat;
	double *d_dfmElmVolVec = fcnObj.d_dfmElmVolVec;

	double *d_vfdVec    = fcnObj.d_vfdVec;
	double *d_sumBufVec = fcnObj.d_sumBufVec;

	varifold(h_vfdPtr,
	         d_dfmLmkPosMat, d_dfmElmVtxMat,
	         d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
	         cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
	         d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
	         d_vfdVec, d_sumBufVec,
	         dfmLmkNum, dfmElmNum, tgtElmNum);

	return;
}

//void varifold(double *h_vfdPtr, double *d_dqVfdMat, double *d_dfmLmkPosMat, fcndata &fcnObj)
//{
//	int dfmLmkNum = fcnObj.prm.vfdLmkNum;
//	int dfmElmNum = fcnObj.prm.vfdElmNum;
//	int tgtElmNum = fcnObj.tgt.vfdElmNum;
//
//	char   cenKnlType  = fcnObj.vfd.cenKnlType;
//	double cenKnlWidth = fcnObj.vfd.cenKnlWidth;
//	char   dirKnlType  = fcnObj.vfd.dirKnlType;
//	double dirKnlWidth = fcnObj.vfd.dirKnlWidth;
//
//	int    *d_dfmElmVtxMat = fcnObj.elm.d_vfdVtxMat;
//	int    *d_dfmElmIfoMat = fcnObj.elm.d_vfdIfoMat;
//	double *d_tgtCenPosMat = fcnObj.tgt.d_cenPosMat;
//	double *d_tgtUniDirMat = fcnObj.tgt.d_uniDirMat;
//	double *d_tgtElmVolVec = fcnObj.tgt.d_elmVolVec;
//	double *d_dfmCenPosMat = fcnObj.d_dfmCenPosMat;
//	double *d_dfmUniDirMat = fcnObj.d_dfmUniDirMat;
//	double *d_dfmElmVolVec = fcnObj.d_dfmElmVolVec;
//
//	double *d_vfdVec    = fcnObj.d_vfdVec;
//	double *d_sumBufVec = fcnObj.d_sumBufVec;
//	double *d_dcVfdMat  = fcnObj.d_dcVfdMat;
//	double *d_dtVfdMat  = fcnObj.d_dtVfdMat;
//
//	varifold(h_vfdPtr, d_dqVfdMat,
//	         d_dfmLmkPosMat, d_dfmElmVtxMat, d_dfmElmIfoMat,
//	         d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
//	         cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
//	         d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
//	         d_vfdVec, d_sumBufVec, d_dcVfdMat, d_dtVfdMat,
//	         dfmLmkNum, dfmElmNum, tgtElmNum);
//
//	return;
//}
