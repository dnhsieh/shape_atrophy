#include "matvec.h"
#include "utility.h"
#include "constants.h"

__global__ void yankVolumeComputeKernel(double *d_exYElmMat, double *d_lmkEdgMat,
                                        double *d_actFcnVec, int *d_elmVtxMat, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec;
		getEdge(q10Vec, q20Vec, d_lmkEdgMat, elmIdx, elmNum);

		double volVal = computeVolume(q10Vec, q20Vec);

		// Q = [q1 - q0, q2 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec);

		// ---

		int q0Idx = d_elmVtxMat[             elmIdx];
		int q1Idx = d_elmVtxMat[    elmNum + elmIdx];
		int q2Idx = d_elmVtxMat[2 * elmNum + elmIdx];

		double f0Val = d_actFcnVec[q0Idx];
		double f1Val = d_actFcnVec[q1Idx];
		double f2Val = d_actFcnVec[q2Idx];
		double fcVal = (f0Val + f1Val + f2Val) / VTXNUM;

		// ---

		vector dv0YVec, dv1YVec, dv2YVec;
		dv0YVec.x = -fcVal * (-QInvMat.x.x - QInvMat.y.x) * volVal;
		dv0YVec.y = -fcVal * (-QInvMat.x.y - QInvMat.y.y) * volVal;
		
		dv1YVec.x = -fcVal *   QInvMat.x.x                * volVal;
		dv1YVec.y = -fcVal *   QInvMat.x.y                * volVal;

		dv2YVec.x = -fcVal *                 QInvMat.y.x  * volVal;
		dv2YVec.y = -fcVal *                 QInvMat.y.y  * volVal;

		setElement(d_exYElmMat, dv0YVec, dv1YVec, dv2YVec, elmIdx, elmNum);
	}

	return;
}

//__global__ void yankBoundaryComputeKernel(double *d_exYBdrMat, double *d_lmkMat,
//                                          double *d_actFcnVec, int *d_bdrVtxMat, int lmkNum, int bdrElmNum)
//{
//	int bdrElmIdx = blockIdx.x * blockDim.x + threadIdx.x;
//	if ( bdrElmIdx < bdrElmNum )
//	{
//		int q0Idx = d_bdrVtxMat[            bdrElmIdx];
//		int q1Idx = d_bdrVtxMat[bdrElmNum + bdrElmIdx];
//
//		vector q0Vec, q1Vec;
//		getVector(q0Vec, d_lmkMat, q0Idx, lmkNum);
//		getVector(q1Vec, d_lmkMat, q1Idx, lmkNum);
//
//		vector tanVec, nmlVec;
//		vectorSubtract(tanVec, q1Vec, q0Vec);
//		nmlVec.x =  tanVec.y;
//		nmlVec.y = -tanVec.x;
//
//		double f0Val = d_actFcnVec[q0Idx];
//		double f1Val = d_actFcnVec[q1Idx];
//
//		// ---	
//	
//		vector dv0YVec, dv1YVec;
//
//		dv0YVec.x = f0Val * nmlVec.x / (VTXNUM - 1.0);
//		dv0YVec.y = f0Val * nmlVec.y / (VTXNUM - 1.0);
//
//		dv1YVec.x = f1Val * nmlVec.x / (VTXNUM - 1.0);
//		dv1YVec.y = f1Val * nmlVec.y / (VTXNUM - 1.0);
//
//		setBoundary(d_exYBdrMat, dv0YVec, dv1YVec, bdrElmIdx, bdrElmNum);
//	}
//
//	return;
//}

__global__ void yankGatherKernel(double *d_exYMat, double *d_exYElmMat, double *d_exYBdrMat,
                                 int *d_vtxElmMat, int *d_vtxBdrMat, int elmNum, int lmkNum, int bdrElmNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		vector exYVec = {0.0, 0.0};

		int adjNum = d_vtxElmMat[lmkIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = d_vtxElmMat[(1 + 2 * adjIdx    ) * lmkNum + lmkIdx];
			int lclIdx = d_vtxElmMat[(1 + 2 * adjIdx + 1) * lmkNum + lmkIdx];

			vector exYElmVec;
			getVector(exYElmVec, d_exYElmMat + lclIdx * elmNum * DIMNUM, elmIdx, elmNum);

			vectorSum(exYVec, exYVec, exYElmVec);
		}

		//adjNum = d_vtxBdrMat[lmkIdx];
		//for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		//{
		//	int bdrElmIdx = d_vtxBdrMat[(1 + 2 * adjIdx    ) * lmkNum + lmkIdx];
		//	int    lclIdx = d_vtxBdrMat[(1 + 2 * adjIdx + 1) * lmkNum + lmkIdx];

		//	vector exYBdrVec;
		//	getVector(exYBdrVec, d_exYBdrMat + lclIdx * bdrElmNum * DIMNUM, bdrElmIdx, bdrElmNum);

		//	vectorSum(exYVec, exYVec, exYBdrVec);
		//}

		setVector(d_exYMat, exYVec, lmkIdx, lmkNum);
	}

	return;
}

void computeExternalYank(double *d_exYMat, double *d_lmkMat, double *d_lmkEdgMat, double *d_actFcnVec,
                         double *d_exYElmMat, double *d_exYBdrMat, int *d_elmVtxMat, int *d_vtxElmMat,
                         int *d_bdrVtxMat, int *d_vtxBdrMat, int lmkNum, int elmNum, int bdrElmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	yankVolumeComputeKernel <<<blkNum, BLKDIM>>> (d_exYElmMat, d_lmkEdgMat, d_actFcnVec,
	                                              d_elmVtxMat, elmNum);

	//blkNum = (bdrElmNum - 1) / BLKDIM + 1;
	//yankBoundaryComputeKernel <<<blkNum, BLKDIM>>> (d_exYBdrMat, d_lmkMat, d_actFcnVec,
	//                                                d_bdrVtxMat, lmkNum, bdrElmNum);
	
	blkNum = (lmkNum - 1) / BLKDIM + 1;
	yankGatherKernel <<<blkNum, BLKDIM>>> (d_exYMat, d_exYElmMat, d_exYBdrMat,
	                                       d_vtxElmMat, d_vtxBdrMat, elmNum, lmkNum, bdrElmNum);

	return;
}
