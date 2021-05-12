#include <cfloat>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "struct.h"
#include "constants.h"

void tauInitial(double *, double *, fcndata &);
void vertexToEdge(double *, double *, int *, int, int);
void vertexToBottom(double *, double *, int *, int, int);
void computeFrame(double *, double *, double *, fcndata &);
void assembleFemMatrix(double *, double *, double *, double *, double *, fcndata &);
void yankTauActivity(double *, double *, fcndata &);
void reactionTauActivity(double *, double *, fcndata &);
void computeKernel(double *, double *, fcndata &);
void addEpsIdentity(double *, double, int);
void cholesky(double *, fcndata &);
void computeExternalYank(double *, double *, double *, double *, fcndata &);
int  ipcg(double *, double *, fcndata &);
void vectoraxpby(double *, double, double *, double, double *, int);
void assembleFemVector(double *, double *, double *, fcndata &);

void deform(double *h_objPtr, double *h_posVec, fcndata &fcnObj)
{
	int     lmkNum = fcnObj.prm.lmkNum;
	int     elmNum = fcnObj.prm.elmNum;
	int  btmElmNum = fcnObj.prm.btmElmNum;
	int    timeNum = fcnObj.prm.timeNum;
	double timeStp = fcnObj.prm.timeStp;

	cudaMemcpy(fcnObj.d_lmkStk, fcnObj.prm.d_lmkIniMat,
	           sizeof(double) * lmkNum * DIMNUM, cudaMemcpyDeviceToDevice);

	cudaMemset(fcnObj.d_vlcStk, 0, sizeof(double) * lmkNum * DIMNUM);
	cudaMemset(fcnObj.d_pnlMat, 0, sizeof(double) * lmkNum * DIMNUM);

	tauInitial(fcnObj.d_tauMat, h_posVec, fcnObj);

	vertexToEdge(fcnObj.d_lmkNowEdgMat, fcnObj.prm.d_lmkIniMat,
	             fcnObj.elm.d_elmVtxMat, lmkNum, elmNum);
	computeFrame(fcnObj.d_nmlNowMat, fcnObj.d_tsvNowMat, fcnObj.prm.d_lmkIniMat, fcnObj);
	assembleFemMatrix(fcnObj.d_ppdNowMat, fcnObj.d_ggdNowMat,
	                  fcnObj.d_lmkNowEdgMat, fcnObj.d_nmlNowMat, fcnObj.d_tsvNowMat, fcnObj);

	for ( int timeIdx = 0; timeIdx < timeNum - 1; ++timeIdx )
	{
		fcnObj.d_lmkNowMat = fcnObj.d_lmkStk +  timeIdx      * lmkNum * DIMNUM;
		fcnObj.d_lmkNxtMat = fcnObj.d_lmkStk + (timeIdx + 1) * lmkNum * DIMNUM;
		fcnObj.d_tauNowVec = fcnObj.d_tauMat +  timeIdx      * lmkNum;
		fcnObj.d_tauNxtVec = fcnObj.d_tauMat + (timeIdx + 1) * lmkNum;
		fcnObj.d_vlcMat    = fcnObj.d_vlcStk +  timeIdx      * lmkNum * DIMNUM;

		vertexToBottom(fcnObj.d_lmkNowBtmMat, fcnObj.d_lmkNowMat,
		               fcnObj.elm.d_btmVtxMat, lmkNum, btmElmNum);

		yankTauActivity(fcnObj.d_ynkActFcnNowVec, fcnObj.d_tauNowVec, fcnObj);
		reactionTauActivity(fcnObj.d_reaActFcnNowVec, fcnObj.d_tauNowVec, fcnObj);

		computeKernel(fcnObj.d_knlMat, fcnObj.d_lmkNowMat, fcnObj); 
		addEpsIdentity(fcnObj.d_knlMat, fcnObj.prm.knlEps, lmkNum);

		cudaMemcpy(fcnObj.d_knLMat, fcnObj.d_knlMat,
		           sizeof(double) * lmkNum * lmkNum, cudaMemcpyDeviceToDevice);
		cholesky(fcnObj.d_knLMat, fcnObj);

		computeExternalYank(fcnObj.d_exYMat, fcnObj.d_lmkNowMat, fcnObj.d_lmkNowEdgMat,
		                    fcnObj.d_ynkActFcnNowVec, fcnObj);

		int cgStatus = ipcg(fcnObj.d_vlcMat, fcnObj.d_exYMat, fcnObj);
		if ( cgStatus != 0 )
		{
			*h_objPtr = DBL_MAX;
			return;
		}

		if ( timeIdx < timeNum - 2 )
		{
			cudaMemcpy(fcnObj.d_vlcMat + lmkNum * DIMNUM, fcnObj.d_vlcMat, 
			           sizeof(double) * lmkNum * DIMNUM, cudaMemcpyDeviceToDevice);
		}

		vectoraxpby(fcnObj.d_lmkNxtMat,
		            1.0, fcnObj.d_lmkNowMat, timeStp, fcnObj.d_vlcMat, lmkNum * DIMNUM);

		vertexToEdge(fcnObj.d_lmkNxtEdgMat, fcnObj.d_lmkNxtMat,
		             fcnObj.elm.d_elmVtxMat, lmkNum, elmNum);
		computeFrame(fcnObj.d_nmlNxtMat, fcnObj.d_tsvNxtMat, fcnObj.d_lmkNxtMat, fcnObj);
		assembleFemMatrix(fcnObj.d_ppdNxtMat, fcnObj.d_ggdNxtMat,
		                  fcnObj.d_lmkNxtEdgMat, fcnObj.d_nmlNxtMat, fcnObj.d_tsvNxtMat, fcnObj);
		assembleFemVector(fcnObj.d_femRpdVec, fcnObj.d_lmkNowEdgMat, fcnObj.d_reaActFcnNowVec, fcnObj);

		vectoraxpby(fcnObj.d_femLftMat, 1.0, fcnObj.d_ppdNxtMat, timeStp, fcnObj.d_ggdNxtMat, lmkNum * lmkNum);
		cudaMemcpy(fcnObj.d_femPpdMat, fcnObj.d_ppdNowMat, sizeof(double) * lmkNum * lmkNum, cudaMemcpyDeviceToDevice);

		double oneVal = 1.0;
		cublasDgemv(fcnObj.blasHdl, CUBLAS_OP_N, lmkNum, lmkNum,
		            &oneVal, fcnObj.d_femPpdMat, lmkNum, fcnObj.d_tauNowVec, 1,
		            &timeStp, fcnObj.d_femRpdVec, 1);

		cusolverDnDpotrf(fcnObj.solvHdl, CUBLAS_FILL_MODE_LOWER, 
		                 lmkNum, fcnObj.d_femLftMat, lmkNum,
		                 fcnObj.d_workspace, fcnObj.h_Lwork, fcnObj.d_status);
		cudaMemcpy(fcnObj.d_tauNxtVec, fcnObj.d_femRpdVec, sizeof(double) * lmkNum, cudaMemcpyDeviceToDevice);
		cusolverDnDpotrs(fcnObj.solvHdl, CUBLAS_FILL_MODE_LOWER, lmkNum, DIMNUM, fcnObj.d_femLftMat, lmkNum,
		                 fcnObj.d_tauNxtVec, lmkNum, fcnObj.d_status);

		cudaMemcpy(fcnObj.d_lmkNowEdgMat, fcnObj.d_lmkNxtEdgMat,
		           sizeof(double) * elmNum * DIMNUM * (VTXNUM - 1), cudaMemcpyDeviceToDevice);
		cudaMemcpy(fcnObj.d_nmlNowMat, fcnObj.d_nmlNxtMat,
		           sizeof(double) * elmNum * DIMNUM, cudaMemcpyDeviceToDevice);
		cudaMemcpy(fcnObj.d_tsvNowMat, fcnObj.d_tsvNxtMat,
		           sizeof(double) * elmNum * DIMNUM, cudaMemcpyDeviceToDevice);
		cudaMemcpy(fcnObj.d_ppdNowMat, fcnObj.d_ppdNxtMat,
		           sizeof(double) * lmkNum * lmkNum, cudaMemcpyDeviceToDevice);
		cudaMemcpy(fcnObj.d_ggdNowMat, fcnObj.d_ggdNxtMat,
		           sizeof(double) * lmkNum * lmkNum, cudaMemcpyDeviceToDevice);
	}

	return;
}
