#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "struct.h"
#include "constants.h"

void assignObjfcnStructMemory(long long &, fcndata &, double *);
void objfcn(double *, double *, fcndata &);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	mxInitGPU();

	fcndata fcnObj = {0};

	mxGPUArray const *lmkIniMat;
	mxGPUArray const *elmVtxMat, *vtxElmMat, *tanVtxMat, *tsvVtxMat, *femVtxMat, *femIfoMat;
	mxGPUArray const *bdrVtxMat, *vtxBdrMat, *btmVtxMat, *vtxBtmMat, *vfdVtxMat;
	mxGPUArray const *tgtCenPosMat, *tgtUniDirMat, *tgtElmVolVec;

	double *h_tauVarMat       = (double *) mxGetDoubles(prhs[ 0]);
	fcnObj.prm.h_tauPrmVec    = (double *) mxGetDoubles(prhs[ 1]);
	lmkIniMat                 =  mxGPUCreateFromMxArray(prhs[ 2]);
	elmVtxMat                 =  mxGPUCreateFromMxArray(prhs[ 3]);
	vtxElmMat                 =  mxGPUCreateFromMxArray(prhs[ 4]);
	tanVtxMat                 =  mxGPUCreateFromMxArray(prhs[ 5]);
	tsvVtxMat                 =  mxGPUCreateFromMxArray(prhs[ 6]);
	bdrVtxMat                 =  mxGPUCreateFromMxArray(prhs[ 7]);
	vtxBdrMat                 =  mxGPUCreateFromMxArray(prhs[ 8]);
	btmVtxMat                 =  mxGPUCreateFromMxArray(prhs[ 9]);
	vtxBtmMat                 =  mxGPUCreateFromMxArray(prhs[10]);
	femVtxMat                 =  mxGPUCreateFromMxArray(prhs[11]);
	femIfoMat                 =  mxGPUCreateFromMxArray(prhs[12]);
	vfdVtxMat                 =  mxGPUCreateFromMxArray(prhs[13]);
	tgtCenPosMat              =  mxGPUCreateFromMxArray(prhs[14]);
	tgtUniDirMat              =  mxGPUCreateFromMxArray(prhs[15]);
	tgtElmVolVec              =  mxGPUCreateFromMxArray(prhs[16]);
	fcnObj.vfd.cenKnlType     =             mxGetScalar(prhs[17]);
	fcnObj.vfd.cenKnlWidth    =             mxGetScalar(prhs[18]);
	fcnObj.vfd.dirKnlType     =             mxGetScalar(prhs[19]);
	fcnObj.vfd.dirKnlWidth    =             mxGetScalar(prhs[20]);
	fcnObj.prm.ldmWgt         =             mxGetScalar(prhs[21]);
	fcnObj.prm.knlOrder       =             mxGetScalar(prhs[22]);
	fcnObj.prm.knlWidth       =             mxGetScalar(prhs[23]);
	fcnObj.prm.knlEps         =             mxGetScalar(prhs[24]);
	fcnObj.prm.h_modVec       = (double *) mxGetDoubles(prhs[25]);
	fcnObj.prm.spdTanVal      =             mxGetScalar(prhs[26]);
	fcnObj.prm.spdTsvVal      =             mxGetScalar(prhs[27]);
	fcnObj.prm.h_ynkActPrmVec = (double *) mxGetDoubles(prhs[28]);
	fcnObj.prm.h_reaActPrmVec = (double *) mxGetDoubles(prhs[29]);
	fcnObj.prm.btmWgt         =             mxGetScalar(prhs[30]);
	fcnObj.prm.timeStp        =             mxGetScalar(prhs[31]);
	fcnObj.prm.timeNum        =             mxGetScalar(prhs[32]);
	fcnObj.pcg.itrMax         =             mxGetScalar(prhs[33]);
	fcnObj.pcg.tolSqu         =             mxGetScalar(prhs[34]);

	int objVarNum = mxGetM(prhs[0]);
	int objTotNum = mxGetN(prhs[0]);

	plhs[0] = mxCreateDoubleMatrix(1, objTotNum, mxREAL);
	double *h_objVec = (double *) mxGetDoubles(plhs[0]);

	// ---

	fcnObj.prm.d_lmkIniMat = (double *) mxGPUGetDataReadOnly(lmkIniMat);
	fcnObj.elm.d_elmVtxMat = (int    *) mxGPUGetDataReadOnly(elmVtxMat);
	fcnObj.elm.d_vtxElmMat = (int    *) mxGPUGetDataReadOnly(vtxElmMat);
	fcnObj.elm.d_tanVtxMat = (int    *) mxGPUGetDataReadOnly(tanVtxMat);
	fcnObj.elm.d_tsvVtxMat = (int    *) mxGPUGetDataReadOnly(tsvVtxMat);
	fcnObj.elm.d_bdrVtxMat = (int    *) mxGPUGetDataReadOnly(bdrVtxMat);
	fcnObj.elm.d_vtxBdrMat = (int    *) mxGPUGetDataReadOnly(vtxBdrMat);
	fcnObj.elm.d_btmVtxMat = (int    *) mxGPUGetDataReadOnly(btmVtxMat);
	fcnObj.elm.d_vtxBtmMat = (int    *) mxGPUGetDataReadOnly(vtxBtmMat);
	fcnObj.elm.d_femVtxMat = (int    *) mxGPUGetDataReadOnly(femVtxMat);
	fcnObj.elm.d_femIfoMat = (int    *) mxGPUGetDataReadOnly(femIfoMat);
	fcnObj.elm.d_vfdVtxMat = (int    *) mxGPUGetDataReadOnly(vfdVtxMat);
	fcnObj.tgt.d_cenPosMat = (double *) mxGPUGetDataReadOnly(tgtCenPosMat);
	fcnObj.tgt.d_uniDirMat = (double *) mxGPUGetDataReadOnly(tgtUniDirMat);
	fcnObj.tgt.d_elmVolVec = (double *) mxGPUGetDataReadOnly(tgtElmVolVec);

	mwSize const *lmkDims    = mxGPUGetDimensions(lmkIniMat);
	mwSize const *elmDims    = mxGPUGetDimensions(elmVtxMat);
	mwSize const *nzrDims    = mxGPUGetDimensions(femVtxMat);
	mwSize const *bdrElmDims = mxGPUGetDimensions(bdrVtxMat);
	mwSize const *btmLmkDims = mxGPUGetDimensions(vtxBtmMat);
	mwSize const *btmElmDims = mxGPUGetDimensions(btmVtxMat);
	mwSize const *vfdElmDims = mxGPUGetDimensions(vfdVtxMat);
	mwSize const *tgtElmDims = mxGPUGetDimensions(tgtCenPosMat);

	fcnObj.prm.varNum    = objVarNum;
	fcnObj.prm.lmkNum    =    lmkDims[0];
	fcnObj.prm.elmNum    =    elmDims[0];
	fcnObj.prm.nzrNum    =    nzrDims[0];
	fcnObj.prm.bdrElmNum = bdrElmDims[0];
	fcnObj.prm.btmLmkNum = btmLmkDims[0];
	fcnObj.prm.btmElmNum = btmElmDims[0];
	fcnObj.prm.vfdLmkNum = fcnObj.prm.lmkNum;
	fcnObj.prm.vfdElmNum = vfdElmDims[0];
	fcnObj.tgt.vfdElmNum = tgtElmDims[0];

	fcnObj.pcg.varNum = fcnObj.prm.lmkNum * DIMNUM;

	// ---

	int lmkNum    = fcnObj.prm.lmkNum;
	int elmNum    = fcnObj.prm.elmNum;
	int bdrElmNum = fcnObj.prm.bdrElmNum;
	int btmElmNum = fcnObj.prm.btmElmNum;
	int vfdElmNum = fcnObj.prm.vfdElmNum;
	int timeNum   = fcnObj.prm.timeNum;

	long long gpuAloDblMemCnt =  fcnObj.pcg.varNum * 6
	                           + lmkNum    * (lmkNum * 8 + DIMNUM * 3 + DIMNUM * timeNum * 2 + timeNum + 3)
	                           + elmNum    * (DIMNUM * 2 + DIMNUM * VTXNUM * 5 + DIMNUM * DIMNUM
	                                          + (1 + VTXNUM) * VTXNUM / 2 + 1)
	                           + bdrElmNum * DIMNUM * (VTXNUM - 1)
	                           + btmElmNum * 2 * DIMNUM * (VTXNUM - 1)
	                           + vfdElmNum * (DIMNUM * 2 + 2) + fcnObj.tgt.vfdElmNum 
	                           + SUMBLKDIM;

	double *gpuDblSpace;
	cudaError_t error = cudaMalloc((void **) &gpuDblSpace, sizeof(double) * gpuAloDblMemCnt);
	if ( error != cudaSuccess )
		mexErrMsgIdAndTxt("objfcn2Dframe:cudaMalloc", "Fail to allocate device memory.");

	cudaMalloc((void **) &(fcnObj.d_status), sizeof(int));

	long long gpuAsgDblMemCnt;
	assignObjfcnStructMemory(gpuAsgDblMemCnt, fcnObj, gpuDblSpace);
	if ( gpuAsgDblMemCnt != gpuAloDblMemCnt )
	{
		mexErrMsgIdAndTxt("objfcn2Dframe:memAssign", 
		                  "Assigned device double memory (%lld) mismatches the allocated memory (%lld).", 
		                  gpuAsgDblMemCnt, gpuAloDblMemCnt);
	}

	// ---

	cublasCreate(&(fcnObj.blasHdl));

	cusolverDnCreate(&(fcnObj.solvHdl));
	cusolverDnDpotrf_bufferSize(fcnObj.solvHdl, CUBLAS_FILL_MODE_LOWER, fcnObj.prm.lmkNum, fcnObj.d_knlMat,
	                            fcnObj.prm.lmkNum, &(fcnObj.h_Lwork));

	cudaMalloc((void **) &(fcnObj.d_workspace), sizeof(double) * fcnObj.h_Lwork);

	// ---

	cudaEvent_t tic, toc;
	float       timeRun;

	cudaEventCreate(&tic);
	cudaEventCreate(&toc);

	// ---

	cudaEventRecord(tic);
	for ( int objIdx = 0; objIdx < objTotNum; ++objIdx )
	{
		double *h_posPtr = h_tauVarMat + objIdx * objVarNum;
		objfcn(h_objVec + objIdx, h_posPtr, fcnObj);
	}
	cudaEventRecord(toc);
	cudaEventSynchronize(toc);
	cudaEventElapsedTime(&timeRun, tic, toc);

	printf("Time %f msec\n", timeRun);

	// ---
	//

	mxGPUDestroyGPUArray(lmkIniMat);
	mxGPUDestroyGPUArray(elmVtxMat);
	mxGPUDestroyGPUArray(vtxElmMat);
	mxGPUDestroyGPUArray(tanVtxMat);
	mxGPUDestroyGPUArray(tsvVtxMat);
	mxGPUDestroyGPUArray(bdrVtxMat);
	mxGPUDestroyGPUArray(vtxBdrMat);
	mxGPUDestroyGPUArray(btmVtxMat);
	mxGPUDestroyGPUArray(vtxBtmMat);
	mxGPUDestroyGPUArray(femVtxMat);
	mxGPUDestroyGPUArray(femIfoMat);
	mxGPUDestroyGPUArray(vfdVtxMat);
	mxGPUDestroyGPUArray(tgtCenPosMat);
	mxGPUDestroyGPUArray(tgtUniDirMat);
	mxGPUDestroyGPUArray(tgtElmVolVec);

	mxFree((void *) lmkDims);
	mxFree((void *) elmDims);
	mxFree((void *) nzrDims);
	mxFree((void *) bdrElmDims);
	mxFree((void *) btmLmkDims);
	mxFree((void *) btmElmDims);
	mxFree((void *) vfdElmDims);
	mxFree((void *) tgtElmDims);

	cudaFree(gpuDblSpace);
	cudaFree(fcnObj.d_status);
	cudaFree(fcnObj.d_workspace);

	cublasDestroy(fcnObj.blasHdl);
	cusolverDnDestroy(fcnObj.solvHdl);

	cudaEventDestroy(tic);
	cudaEventDestroy(toc);

	return;
}

