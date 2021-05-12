#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void femVecComputeKernel(double *d_rpdElmMat, double *d_lmkEdgMat, double *d_actFcnVec,
                                    int *d_elmVtxMat, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec, q30Vec;
		getEdge(q10Vec, q20Vec, q30Vec, d_lmkEdgMat, elmIdx, elmNum);

		double absDetQ = fabs(det(q10Vec, q20Vec, q30Vec));

		int q0Idx = d_elmVtxMat[             elmIdx];
		int q1Idx = d_elmVtxMat[    elmNum + elmIdx];
		int q2Idx = d_elmVtxMat[2 * elmNum + elmIdx];
		int q3Idx = d_elmVtxMat[3 * elmNum + elmIdx];

		double f0Val = d_actFcnVec[q0Idx];
		double f1Val = d_actFcnVec[q1Idx];
		double f2Val = d_actFcnVec[q2Idx];
		double f3Val = d_actFcnVec[q3Idx];

		d_rpdElmMat[             elmIdx] = (2.0 * f0Val +       f1Val +       f2Val +       f3Val) * absDetQ / 120.0;
		d_rpdElmMat[    elmNum + elmIdx] = (      f0Val + 2.0 * f1Val +       f2Val +       f3Val) * absDetQ / 120.0;
		d_rpdElmMat[2 * elmNum + elmIdx] = (      f0Val +       f1Val + 2.0 * f2Val +       f3Val) * absDetQ / 120.0;
		d_rpdElmMat[3 * elmNum + elmIdx] = (      f0Val +       f1Val +       f2Val + 2.0 * f3Val) * absDetQ / 120.0;
	}
	
	return;
}

__global__ void femVecGatherKernel(double *d_rpdVec, double *d_rpdElmMat,
                                   int *d_vtxElmMat, int elmNum, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		double rpdVal = 0.0;

		int adjNum = d_vtxElmMat[lmkIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = d_vtxElmMat[(1 + 2 * adjIdx    ) * lmkNum + lmkIdx];
			int lclIdx = d_vtxElmMat[(1 + 2 * adjIdx + 1) * lmkNum + lmkIdx];

			double rpdElmVal = d_rpdElmMat[lclIdx * elmNum + elmIdx];
			rpdVal += rpdElmVal;
		}

		d_rpdVec[lmkIdx] = rpdVal;
	}

	return;
}

void assembleFemVector(double *d_rpdVec, double *d_lmkEdgMat, double *d_actFcnVec, double *d_rpdElmMat,
                       int *d_elmVtxMat, int *d_vtxElmMat, int lmkNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	femVecComputeKernel <<<blkNum, BLKDIM>>> (d_rpdElmMat, d_lmkEdgMat, d_actFcnVec,
	                                          d_elmVtxMat, elmNum);

	blkNum = (lmkNum - 1) / BLKDIM + 1;
	femVecGatherKernel <<<blkNum, BLKDIM>>> (d_rpdVec, d_rpdElmMat, d_vtxElmMat, elmNum, lmkNum);

	return;
}
