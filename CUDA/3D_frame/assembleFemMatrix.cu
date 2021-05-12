#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void femMatComputeKernel(double *d_ppdElmVec, double *d_ggdElmMat, 
                                    double *d_lmkEdgMat, double *d_nmlMat, double *d_tsvMat,
                                    double spdTanVal, double spdTsvVal, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec, q30Vec;
		getEdge(q10Vec, q20Vec, q30Vec, d_lmkEdgMat, elmIdx, elmNum);

		double absDetQ = fabs(det(q10Vec, q20Vec, q30Vec));

		// Q = [q1 - q0, q2 - q0, q3 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec, q30Vec);

		vector nmlVec, tsvVec;
		getVector(nmlVec, d_nmlMat, elmIdx, elmNum);
		getVector(tsvVec, d_tsvMat, elmIdx, elmNum);

		vector QInvNmlVec, QInvTsvVec;
		matVecMul(QInvNmlVec, QInvMat, nmlVec);
		matVecMul(QInvTsvVec, QInvMat, tsvVec);

		double e1G0Val = -QInvMat.x.x - QInvMat.y.x - QInvMat.z.x;
		double e1G1Val =  QInvMat.x.x                            ;
		double e1G2Val =                QInvMat.y.x              ;
		double e1G3Val =                              QInvMat.z.x;

		double e2G0Val = -QInvMat.x.y - QInvMat.y.y - QInvMat.z.y;
		double e2G1Val =  QInvMat.x.y                            ;
		double e2G2Val =                QInvMat.y.y              ;
		double e2G3Val =                              QInvMat.z.y;

		double e3G0Val = -QInvMat.x.z - QInvMat.y.z - QInvMat.z.z;
		double e3G1Val =  QInvMat.x.z                            ;
		double e3G2Val =                QInvMat.y.z              ;
		double e3G3Val =                              QInvMat.z.z;

		double nmlG0Val = -QInvNmlVec.x - QInvNmlVec.y - QInvNmlVec.z;
		double nmlG1Val =  QInvNmlVec.x                              ;
		double nmlG2Val =                 QInvNmlVec.y               ;
		double nmlG3Val =                                QInvNmlVec.z;

		double tsvG0Val = -QInvTsvVec.x - QInvTsvVec.y - QInvTsvVec.z;
		double tsvG1Val =  QInvTsvVec.x                              ;
		double tsvG2Val =                 QInvTsvVec.y               ;
		double tsvG3Val =                                QInvTsvVec.z;

		d_ppdElmVec[elmIdx] = absDetQ / 120.0;

		d_ggdElmMat[             elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G0Val *  e1G0Val
		                                                                   +  e2G0Val *  e2G0Val
		                                                                   +  e3G0Val *  e3G0Val
		                                                                   - nmlG0Val * nmlG0Val )
		                                                    + spdTsvVal * tsvG0Val * tsvG0Val      );

		d_ggdElmMat[    elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G0Val *  e1G1Val
		                                                                   +  e2G0Val *  e2G1Val
		                                                                   +  e3G0Val *  e3G1Val
		                                                                   - nmlG0Val * nmlG1Val )
		                                                    + spdTsvVal * tsvG0Val * tsvG1Val      );

		d_ggdElmMat[2 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G0Val *  e1G2Val
		                                                                   +  e2G0Val *  e2G2Val
		                                                                   +  e3G0Val *  e3G2Val
		                                                                   - nmlG0Val * nmlG2Val )
		                                                    + spdTsvVal * tsvG0Val * tsvG2Val );

		d_ggdElmMat[3 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G0Val *  e1G3Val
		                                                                   +  e2G0Val *  e2G3Val
		                                                                   +  e3G0Val *  e3G3Val
		                                                                   - nmlG0Val * nmlG3Val )
		                                                    + spdTsvVal * tsvG0Val * tsvG3Val );

		d_ggdElmMat[4 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G1Val *  e1G1Val
		                                                                   +  e2G1Val *  e2G1Val
		                                                                   +  e3G1Val *  e3G1Val
		                                                                   - nmlG1Val * nmlG1Val )
		                                                    + spdTsvVal * tsvG1Val * tsvG1Val );

		d_ggdElmMat[5 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G1Val *  e1G2Val
		                                                                   +  e2G1Val *  e2G2Val
		                                                                   +  e3G1Val *  e3G2Val
		                                                                   - nmlG1Val * nmlG2Val )
		                                                    + spdTsvVal * tsvG1Val * tsvG2Val );

		d_ggdElmMat[6 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G1Val *  e1G3Val
		                                                                   +  e2G1Val *  e2G3Val
		                                                                   +  e3G1Val *  e3G3Val
		                                                                   - nmlG1Val * nmlG3Val )
		                                                    + spdTsvVal * tsvG1Val * tsvG3Val );

		d_ggdElmMat[7 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G2Val *  e1G2Val
		                                                                   +  e2G2Val *  e2G2Val
		                                                                   +  e3G2Val *  e3G2Val
		                                                                   - nmlG2Val * nmlG2Val )
		                                                    + spdTsvVal * tsvG2Val * tsvG2Val );

		d_ggdElmMat[8 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G2Val *  e1G3Val
		                                                                   +  e2G2Val *  e2G3Val
		                                                                   +  e3G2Val *  e3G3Val
		                                                                   - nmlG2Val * nmlG3Val )
		                                                    + spdTsvVal * tsvG2Val * tsvG3Val );

		d_ggdElmMat[9 * elmNum + elmIdx] = absDetQ / 6.0 * (  spdTanVal * (   e1G3Val *  e1G3Val
		                                                                   +  e2G3Val *  e2G3Val
		                                                                   +  e3G3Val *  e3G3Val
		                                                                   - nmlG3Val * nmlG3Val )
		                                                    + spdTsvVal * tsvG3Val * tsvG3Val );
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
                       double *d_lmkEdgMat, double *d_nmlMat, double *d_tsvMat,
                       double spdTanVal, double spdTsvVal, double *d_ppdElmVec, double *d_ggdElmMat,
                       int *d_femVtxMat, int *d_femIfoMat, int lmkNum, int nzrNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	femMatComputeKernel <<<blkNum, BLKDIM>>> (d_ppdElmVec, d_ggdElmMat, d_lmkEdgMat,
	                                          d_nmlMat, d_tsvMat, spdTanVal, spdTsvVal, elmNum);

	blkNum = (nzrNum - 1) / BLKDIM + 1;
	cudaMemset(d_ppdMat, 0, sizeof(double) * lmkNum * lmkNum);
	cudaMemset(d_ggdMat, 0, sizeof(double) * lmkNum * lmkNum);
	femMatGatherKernel <<<blkNum, BLKDIM>>> (d_ppdMat, d_ggdMat, d_ppdElmVec, d_ggdElmMat,
	                                         d_femVtxMat, d_femIfoMat, lmkNum, nzrNum);

	return;
}
