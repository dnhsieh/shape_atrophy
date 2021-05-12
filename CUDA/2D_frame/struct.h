#ifndef STRUCT_H
#define STRUCT_H

#include <cusolverDn.h>
#include <cublas_v2.h>

struct parameters
{
	int     varNum;

	int     lmkNum;
	int     elmNum;
	int     nzrNum;
	int     bdrElmNum;
	int     btmLmkNum;
	int     btmElmNum;
	int     vfdLmkNum;
	int     vfdElmNum;

	double *h_tauPrmVec;
	double *d_lmkIniMat;
	double  ldmWgt;
	int     knlOrder;
	double  knlWidth;
	double  knlEps;
	double *h_modVec;
	double  spdTanVal;
	double  spdTsvVal;
	double *h_ynkActPrmVec;
	double *h_reaActPrmVec;
	double  btmWgt;
	double  timeStp;
	int     timeNum;
};

struct element
{
	int *d_elmVtxMat;
	int *d_vtxElmMat;
	int *d_tanVtxMat;
	int *d_tsvVtxMat;
	int *d_bdrVtxMat;
	int *d_vtxBdrMat;
	int *d_btmVtxMat;
	int *d_vtxBtmMat;
	int *d_femVtxMat;
	int *d_femIfoMat;
	int *d_vfdVtxMat;
};

struct target
{
	int vfdElmNum;

	double *d_cenPosMat;
	double *d_uniDirMat;
	double *d_elmVolVec;
};

struct varifold
{
	char   cenKnlType;
	double cenKnlWidth;
	char   dirKnlType;
	double dirKnlWidth;
};

struct pcgdata
{
	int    varNum;
	int    itrMax;
	double tolSqu;

	double *d_AdVec;   // pcgVarNum
	double *d_BdVec;   // pcgVarNum
	double *d_KidVec;  // pcgVarNum
	double *d_rVec;    // pcgVarNum
	double *d_KrVec;   // pcgVarNum
	double *d_dVec;    // pcgVarNum
};

struct fcndata
{
	struct parameters prm;
	struct element    elm;
	struct target     tgt;
	struct varifold   vfd;
	struct pcgdata    pcg;

	double *d_lmkStk;          // lmkNum * DIMNUM * timeNum
	double *d_lmkNowMat;       // no memory allocation, pointing to the data
	double *d_lmkNxtMat;       // no memory allocation, pointing to the data
	double *d_lmkNowEdgMat;    // elmNum * DIMNUM * (VTXNUM - 1)
	double *d_lmkNxtEdgMat;    // elmNum * DIMNUM * (VTXNUM - 1)
	double *d_lmkNowBtmMat;    // btmElmNum * DIMNUM * (VTXNUM - 1)
	double *d_tauMat;          // lmkNum * timeNum
	double *d_tauNowVec;       // no memory allocation, pointing to the data
	double *d_tauNxtVec;       // no memory allocation, pointing to the data
	double *d_vlcStk;          // lmkNum * DIMNUM * (timeNum - 1)
	double *d_vlcMat;          // no memory allocation, pointing to the data
	double *d_tanNowMat;       // elmNum * DIMNUM
	double *d_tanNxtMat;       // elmNum * DIMNUM
	double *d_tsvNowMat;       // elmNum * DIMNUM
	double *d_tsvNxtMat;       // elmNum * DIMNUM
	double *d_ppdNowMat;       // lmkNum * lmkNum 
	double *d_ppdNxtMat;       // lmkNum * lmkNum 
	double *d_ggdNowMat;       // lmkNum * lmkNum 
	double *d_ggdNxtMat;       // lmkNum * lmkNum 
	double *d_femLftMat;       // lmkNum * lmkNum 
	double *d_femPpdMat;       // lmkNum * lmkNum 
	double *d_femRpdVec;       // lmkNum
	double *d_ynkActFcnNowVec; // lmkNum
	double *d_reaActFcnNowVec; // lmkNum
	double *d_knlMat;          // lmkNum * lmkNum
	double *d_knLMat;          // lmkNum * lmkNum 
	double *d_exYMat;          // lmkNum * DIMNUM
	double *d_epvMat;          // elmNum * DIMNUM * DIMNUM
	double *d_elYMat;          // lmkNum * DIMNUM
	double *d_pnlMat;          // lmkNum * DIMNUM

	double *d_exYElmMat;       // elmNum * DIMNUM * VTXNUM
	double *d_exYBdrMat;       // bdrElmNum * DIMNUM * (VTXNUM - 1)
	double *d_KivMat;          // lmkNum * DIMNUM
	double *d_elYElmMat;       // elmNum * DIMNUM * VTXNUM
	double *d_pnlBtmMat;       // btmElmNum * DIMNUM * (VTXNUM - 1)
	double *d_ppdElmVec;       // elmNum
	double *d_ggdElmMat;       // elmNum * (1 + VTXNUM) * VTXNUM / 2
	double *d_rpdElmMat;       // elmNum * DIMNUM * VTXNUM
	double *d_dfmCenPosMat;    // vfdElmNum * DIMNUM
	double *d_dfmUniDirMat;    // vfdElmNum * DIMNUM
	double *d_dfmElmVolVec;    // vfdElmNum
	double *d_vfdVec;          // vfdElmNum + tgtElmNum
	double *d_sumBufVec;       // SUMBLKDIM

	cublasHandle_t     blasHdl;
	cusolverDnHandle_t solvHdl;

	int     h_Lwork;
	double *d_workspace;       // sizeof(double) * h_Lwork
	int    *d_status;          // sizeof(int) * 1
};

struct optdata
{
	int    itrMax;
	double tolVal;
	double wolfe1;
	double wolfe2;
	bool   verbose;

	double *h_apHMat;   // optVarNum * optVarNum
	double *h_dirVec;   // optVarNum
	double *h_posNxt;   // optVarNum
	double *h_grdNxt;   // optVarNum
	double *h_dspVec;   // optVarNum
	double *h_dgdVec;   // optVarNum
	double *h_tmpVec;   // optVarNum
};

#endif
