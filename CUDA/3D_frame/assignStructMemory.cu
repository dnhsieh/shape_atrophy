#include <cstdlib>
#include "constants.h"
#include "struct.h"

void assignOptStructMemory(optdata &optObj, double *optSpace, int optVarNum)
{
	double *h_dblPtr = optSpace;

	optObj.h_apHMat = h_dblPtr;
	h_dblPtr += optVarNum * optVarNum;

	optObj.h_dirVec = h_dblPtr;
	h_dblPtr += optVarNum;

	optObj.h_posNxt = h_dblPtr;
	h_dblPtr += optVarNum;

	optObj.h_grdNxt = h_dblPtr;
	h_dblPtr += optVarNum;

	optObj.h_dspVec = h_dblPtr;
	h_dblPtr += optVarNum;

	optObj.h_dgdVec = h_dblPtr;
	h_dblPtr += optVarNum;

	optObj.h_tmpVec = h_dblPtr;
	h_dblPtr += optVarNum;

	return;
}

void assignObjfcnStructMemory(long long &gpuDblMemCnt, fcndata &fcnObj, double *gpuDblSpace)
{
	gpuDblMemCnt = 0;

	double *d_dblPtr = gpuDblSpace;

	int    lmkNum = fcnObj.prm.lmkNum;
	int    elmNum = fcnObj.prm.elmNum;
	int btmElmNum = fcnObj.prm.btmElmNum;
	int bdrElmNum = fcnObj.prm.bdrElmNum;
	int vfdElmNum = fcnObj.prm.vfdElmNum;
	int   timeNum = fcnObj.prm.timeNum;

	// ---

	int pcgVarNum = fcnObj.pcg.varNum;

	fcnObj.pcg.d_AdVec = d_dblPtr;
	d_dblPtr     += pcgVarNum;
	gpuDblMemCnt += pcgVarNum;

	fcnObj.pcg.d_BdVec = d_dblPtr;
	d_dblPtr     += pcgVarNum;
	gpuDblMemCnt += pcgVarNum;

	fcnObj.pcg.d_KidVec = d_dblPtr;
	d_dblPtr     += pcgVarNum;
	gpuDblMemCnt += pcgVarNum;

	fcnObj.pcg.d_rVec = d_dblPtr;
	d_dblPtr     += pcgVarNum;
	gpuDblMemCnt += pcgVarNum;

	fcnObj.pcg.d_KrVec = d_dblPtr;
	d_dblPtr     += pcgVarNum;
	gpuDblMemCnt += pcgVarNum;

	fcnObj.pcg.d_dVec = d_dblPtr;
	d_dblPtr     += pcgVarNum;
	gpuDblMemCnt += pcgVarNum;

	// ---

	fcnObj.d_lmkStk = d_dblPtr;
	d_dblPtr     += lmkNum * DIMNUM * timeNum;
	gpuDblMemCnt += lmkNum * DIMNUM * timeNum;

	fcnObj.d_lmkNowEdgMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM * (VTXNUM - 1);
	gpuDblMemCnt += elmNum * DIMNUM * (VTXNUM - 1);

	fcnObj.d_lmkNxtEdgMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM * (VTXNUM - 1);
	gpuDblMemCnt += elmNum * DIMNUM * (VTXNUM - 1);

	fcnObj.d_lmkNowBtmMat = d_dblPtr;
	d_dblPtr     += btmElmNum * DIMNUM * (VTXNUM - 1);
	gpuDblMemCnt += btmElmNum * DIMNUM * (VTXNUM - 1);

	fcnObj.d_tauMat = d_dblPtr;
	d_dblPtr     += lmkNum * timeNum;
	gpuDblMemCnt += lmkNum * timeNum;

	fcnObj.d_vlcStk = d_dblPtr;
	d_dblPtr     += lmkNum * DIMNUM * (timeNum - 1);
	gpuDblMemCnt += lmkNum * DIMNUM * (timeNum - 1);

	fcnObj.d_nmlNowMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM;
	gpuDblMemCnt += elmNum * DIMNUM;

	fcnObj.d_nmlNxtMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM;
	gpuDblMemCnt += elmNum * DIMNUM;

	fcnObj.d_tsvNowMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM;
	gpuDblMemCnt += elmNum * DIMNUM;

	fcnObj.d_tsvNxtMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM;
	gpuDblMemCnt += elmNum * DIMNUM;

	fcnObj.d_ppdElmVec = d_dblPtr;
	d_dblPtr     += elmNum;
	gpuDblMemCnt += elmNum;

	fcnObj.d_ppdNowMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_ppdNxtMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_ggdElmMat = d_dblPtr;
	d_dblPtr     += elmNum * (VTXNUM + 1) * VTXNUM / 2;
	gpuDblMemCnt += elmNum * (VTXNUM + 1) * VTXNUM / 2;

	fcnObj.d_ggdNowMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_ggdNxtMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_ynkActFcnNowVec = d_dblPtr;
	d_dblPtr     += lmkNum;
	gpuDblMemCnt += lmkNum;

	fcnObj.d_reaActFcnNowVec = d_dblPtr;
	d_dblPtr     += lmkNum;
	gpuDblMemCnt += lmkNum;

	fcnObj.d_knlMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_knLMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_exYElmMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM * VTXNUM;
	gpuDblMemCnt += elmNum * DIMNUM * VTXNUM;

	fcnObj.d_exYBdrMat = d_dblPtr;
	d_dblPtr     += bdrElmNum * DIMNUM * (VTXNUM - 1);
	gpuDblMemCnt += bdrElmNum * DIMNUM * (VTXNUM - 1);

	fcnObj.d_exYMat = d_dblPtr;
	d_dblPtr     += lmkNum * DIMNUM;
	gpuDblMemCnt += lmkNum * DIMNUM;

	fcnObj.d_KivMat = d_dblPtr;
	d_dblPtr     += lmkNum * DIMNUM;
	gpuDblMemCnt += lmkNum * DIMNUM;

	fcnObj.d_epvMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM * DIMNUM;
	gpuDblMemCnt += elmNum * DIMNUM * DIMNUM;

	fcnObj.d_elYElmMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM * VTXNUM;
	gpuDblMemCnt += elmNum * DIMNUM * VTXNUM;

	fcnObj.d_elYMat = d_dblPtr;
	d_dblPtr     += lmkNum * DIMNUM;
	gpuDblMemCnt += lmkNum * DIMNUM;

	fcnObj.d_pnlBtmMat = d_dblPtr;
	d_dblPtr     += btmElmNum * DIMNUM * (VTXNUM - 1);
	gpuDblMemCnt += btmElmNum * DIMNUM * (VTXNUM - 1);

	fcnObj.d_pnlMat = d_dblPtr;
	d_dblPtr     += lmkNum * DIMNUM;
	gpuDblMemCnt += lmkNum * DIMNUM;

	fcnObj.d_femLftMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_femPpdMat = d_dblPtr;
	d_dblPtr     += lmkNum * lmkNum;
	gpuDblMemCnt += lmkNum * lmkNum;

	fcnObj.d_rpdElmMat = d_dblPtr;
	d_dblPtr     += elmNum * DIMNUM * VTXNUM;
	gpuDblMemCnt += elmNum * DIMNUM * VTXNUM;

	fcnObj.d_femRpdVec = d_dblPtr;
	d_dblPtr     += lmkNum;
	gpuDblMemCnt += lmkNum;

	fcnObj.d_dfmCenPosMat = d_dblPtr;
	d_dblPtr     += vfdElmNum * DIMNUM;
	gpuDblMemCnt += vfdElmNum * DIMNUM;

	fcnObj.d_dfmUniDirMat = d_dblPtr;
	d_dblPtr     += vfdElmNum * DIMNUM;
	gpuDblMemCnt += vfdElmNum * DIMNUM;

	fcnObj.d_dfmElmVolVec = d_dblPtr;
	d_dblPtr     += vfdElmNum;
	gpuDblMemCnt += vfdElmNum;

	fcnObj.d_vfdVec = d_dblPtr;
	d_dblPtr     += vfdElmNum + fcnObj.tgt.vfdElmNum;
	gpuDblMemCnt += vfdElmNum + fcnObj.tgt.vfdElmNum;

	fcnObj.d_sumBufVec = d_dblPtr;
	d_dblPtr     += SUMBLKDIM;
	gpuDblMemCnt += SUMBLKDIM;

	return;
}
