#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void femMatComputeKernel(double *d_ppdElmVec, double *d_ggdElmMat, 
                                    double *d_lmkEdgMat, double *d_tanMat, double *d_tsvMat,
                                    double spdTanVal, double spdTsvVal, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec;
		getEdge(q10Vec, q20Vec, d_lmkEdgMat, elmIdx, elmNum);

		double absDetQ = fabs(det(q10Vec, q20Vec));

		// Q = [q1 - q0, q2 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec);

		vector tanVec, tsvVec;
		getVector(tanVec, d_tanMat, elmIdx, elmNum);
		getVector(tsvVec, d_tsvMat, elmIdx, elmNum);

		vector QInvTanVec, QInvTsvVec;
		matVecMul(QInvTanVec, QInvMat, tanVec);
		matVecMul(QInvTsvVec, QInvMat, tsvVec);

		double tanG0Val = -QInvTanVec.x - QInvTanVec.y;
		double tanG1Val =  QInvTanVec.x               ;
		double tanG2Val =                 QInvTanVec.y;

		double tsvG0Val = -QInvTsvVec.x - QInvTsvVec.y;
		double tsvG1Val =  QInvTsvVec.x               ;
		double tsvG2Val =                 QInvTsvVec.y;

		d_ppdElmVec[elmIdx] = absDetQ / 24.0;

		d_ggdElmMat[             elmIdx] = 0.5 * absDetQ * (  spdTanVal * tanG0Val * tanG0Val
		                                                    + spdTsvVal * tsvG0Val * tsvG0Val );

		d_ggdElmMat[    elmNum + elmIdx] = 0.5 * absDetQ * (  spdTanVal * tanG0Val * tanG1Val
                                                          + spdTsvVal * tsvG0Val * tsvG1Val );

		d_ggdElmMat[2 * elmNum + elmIdx] = 0.5 * absDetQ * (  spdTanVal * tanG0Val * tanG2Val
                                                          + spdTsvVal * tsvG0Val * tsvG2Val );

		d_ggdElmMat[3 * elmNum + elmIdx] = 0.5 * absDetQ * (  spdTanVal * tanG1Val * tanG1Val
                                                          + spdTsvVal * tsvG1Val * tsvG1Val );

		d_ggdElmMat[4 * elmNum + elmIdx] = 0.5 * absDetQ * (  spdTanVal * tanG1Val * tanG2Val
                                                          + spdTsvVal * tsvG1Val * tsvG2Val );

		d_ggdElmMat[5 * elmNum + elmIdx] = 0.5 * absDetQ * (  spdTanVal * tanG2Val * tanG2Val
                                                          + spdTsvVal * tsvG2Val * tsvG2Val );
	}
	
	return;
}

__global__ void femMatGatherKernel(double *d_ppdMat, double *d_ggdMat,
                                   double *d_ppdElmVec, double *d_ggdElmMat,
                                   int *d_femVtxMat, int *d_femIfoMat, int lmkNum, int nzrNum)
{
	int nzrIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( nzrIdx < nzrNum )
	{
		int rowIdx = d_femVtxMat[         nzrIdx];
		int colIdx = d_femVtxMat[nzrNum + nzrIdx];

		double ppdVal = 0.0, ggdVal = 0.0;

		int    adjNum = d_femIfoMat[         nzrIdx];	
		double mulVal = d_femIfoMat[nzrNum + nzrIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = d_femIfoMat[(2 + 2 * adjIdx    ) * nzrNum + nzrIdx];
			int ggdIdx = d_femIfoMat[(2 + 2 * adjIdx + 1) * nzrNum + nzrIdx];

			double ppdElmVal = d_ppdElmVec[elmIdx];
			double ggdElmVal = d_ggdElmMat[ggdIdx];

			ppdVal += mulVal * ppdElmVal;
			ggdVal += ggdElmVal;
		}

		d_ppdMat[colIdx * lmkNum + rowIdx] = ppdVal;
		d_ggdMat[colIdx * lmkNum + rowIdx] = ggdVal;
	}

	return;
}

void assembleFemMatrix(double *d_ppdMat, double *d_ggdMat,
                       double *d_lmkEdgMat, double *d_tanMat, double *d_tsvMat,
                       double spdTanVal, double spdTsvVal, double *d_ppdElmVec, double *d_ggdElmMat,
                       int *d_femVtxMat, int *d_femIfoMat, int lmkNum, int nzrNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	femMatComputeKernel <<<blkNum, BLKDIM>>> (d_ppdElmVec, d_ggdElmMat, d_lmkEdgMat,
	                                          d_tanMat, d_tsvMat, spdTanVal, spdTsvVal, elmNum);

	blkNum = (nzrNum - 1) / BLKDIM + 1;
	cudaMemset(d_ppdMat, 0, sizeof(double) * lmkNum * lmkNum);
	cudaMemset(d_ggdMat, 0, sizeof(double) * lmkNum * lmkNum);
	femMatGatherKernel <<<blkNum, BLKDIM>>> (d_ppdMat, d_ggdMat, d_ppdElmVec, d_ggdElmMat,
	                                         d_femVtxMat, d_femIfoMat, lmkNum, nzrNum);

	return;
}
