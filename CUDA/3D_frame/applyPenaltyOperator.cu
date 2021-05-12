#include "matvec.h"
#include "constants.h"

__global__ void penaltyComputeKernel(double *d_pnlBtmMat, double *d_lmkBtmMat, double *d_vlcMat,
                                     int *d_btmVtxMat, int lmkNum, int btmElmNum)
{
	int btmElmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( btmElmIdx < btmElmNum )
	{
		vector q0Vec, q1Vec, q2Vec;
		getBoundary(q0Vec, q1Vec, q2Vec, d_lmkBtmMat, btmElmIdx, btmElmNum);

		double triArea = 0.5 * fabs(  (q1Vec.x - q0Vec.x) * (q2Vec.y - q0Vec.y)
		                            - (q1Vec.y - q0Vec.y) * (q2Vec.x - q0Vec.x));

		int q0Idx = d_btmVtxMat[                btmElmIdx];
		int q1Idx = d_btmVtxMat[    btmElmNum + btmElmIdx];
		int q2Idx = d_btmVtxMat[2 * btmElmNum + btmElmIdx];

		vector v0Vec, v1Vec, v2Vec;
		getVector(v0Vec, d_vlcMat, q0Idx, lmkNum);
		getVector(v1Vec, d_vlcMat, q1Idx, lmkNum);
		getVector(v2Vec, d_vlcMat, q2Idx, lmkNum);

		vector dv0PVec, dv1PVec, dv2PVec;

		dv0PVec.x = v0Vec.x * triArea / (VTXNUM - 1.0);
		dv0PVec.y = v0Vec.y * triArea / (VTXNUM - 1.0);
		dv0PVec.z = v0Vec.z * triArea / (VTXNUM - 1.0);

		dv1PVec.x = v1Vec.x * triArea / (VTXNUM - 1.0);
		dv1PVec.y = v1Vec.y * triArea / (VTXNUM - 1.0);
		dv1PVec.z = v1Vec.z * triArea / (VTXNUM - 1.0);

		dv2PVec.x = v2Vec.x * triArea / (VTXNUM - 1.0);
		dv2PVec.y = v2Vec.y * triArea / (VTXNUM - 1.0);
		dv2PVec.z = v2Vec.z * triArea / (VTXNUM - 1.0);

		setBoundary(d_pnlBtmMat, dv0PVec, dv1PVec, dv2PVec, btmElmIdx, btmElmNum);
	}

	return;
}

__global__ void penaltyGatherKernel(double *d_pnlMat, double *d_pnlBtmMat, int *d_vtxBtmMat, 
                                    int btmElmNum, int lmkNum, int btmLmkNum)
{
	int btmLmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( btmLmkIdx < btmLmkNum )
	{
		vector pnlVec = {0.0, 0.0, 0.0};

		int adjNum = d_vtxBtmMat[btmLmkIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int btmElmIdx = d_vtxBtmMat[(1 + 2 * adjIdx    ) * btmLmkNum + btmLmkIdx];
			int    lclIdx = d_vtxBtmMat[(1 + 2 * adjIdx + 1) * btmLmkNum + btmLmkIdx];

			vector pnlBtmVec;
			getVector(pnlBtmVec, d_pnlBtmMat + lclIdx * btmElmNum * DIMNUM, btmElmIdx, btmElmNum);

			vectorSum(pnlVec, pnlVec, pnlBtmVec);
		}

		setVector(d_pnlMat, pnlVec, btmLmkIdx, lmkNum);
	}

	return;
}

void applyPenaltyOperator(double *d_pnlMat, double *d_lmkBtmMat, double *d_vlcMat,
                          double *d_pnlBtmMat, int *d_btmVtxMat, int *d_vtxBtmMat,
                          int lmkNum, int btmElmNum, int btmLmkNum) 
{
	int blkNum = (btmElmNum - 1) / BLKDIM + 1;
	penaltyComputeKernel <<<blkNum, BLKDIM>>> (d_pnlBtmMat, d_lmkBtmMat, d_vlcMat,
	                                           d_btmVtxMat, lmkNum, btmElmNum);

	blkNum = (btmLmkNum - 1) / BLKDIM + 1;
	cudaMemset(d_pnlMat, 0, sizeof(double) * lmkNum * DIMNUM);
	penaltyGatherKernel <<<blkNum, BLKDIM>>> (d_pnlMat, d_pnlBtmMat,
	                                          d_vtxBtmMat, btmElmNum, lmkNum, btmLmkNum);

	return;
}
