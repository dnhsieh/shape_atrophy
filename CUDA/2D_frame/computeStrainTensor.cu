#include "matvec.h"
#include "constants.h"

__global__ void strainKernel(double *d_epvMat, double *d_lmkEdgMat, double *d_vlcMat, int *d_elmVtxMat, 
                             int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec;
		getEdge(q10Vec, q20Vec, d_lmkEdgMat, elmIdx, elmNum);

		int q0Idx = d_elmVtxMat[             elmIdx];
		int q1Idx = d_elmVtxMat[    elmNum + elmIdx];
		int q2Idx = d_elmVtxMat[2 * elmNum + elmIdx];

		vector v0Vec, v1Vec, v2Vec;
		getVector(v0Vec, d_vlcMat, q0Idx, lmkNum);
		getVector(v1Vec, d_vlcMat, q1Idx, lmkNum);
		getVector(v2Vec, d_vlcMat, q2Idx, lmkNum);
	
		vector v10Vec, v20Vec;
		vectorSubtract(v10Vec, v1Vec, v0Vec);
		vectorSubtract(v20Vec, v2Vec, v0Vec);

		// Q = [q1 - q0, q2 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec);

		// V = [v1 - v0, v2 - v0]
		matrix VMat;
		columns2Mat(VMat, v10Vec, v20Vec);

		matrix DvMat;
		matMatMul(DvMat, VMat, QInvMat);

		d_epvMat[             elmIdx] = DvMat.x.x;
		d_epvMat[    elmNum + elmIdx] = 0.5 * (DvMat.x.y + DvMat.y.x);
		d_epvMat[2 * elmNum + elmIdx] = 0.5 * (DvMat.x.y + DvMat.y.x);
		d_epvMat[3 * elmNum + elmIdx] = DvMat.y.y;
	}

	return;
}

void computeStrainTensor(double *d_epvMat, double *d_lmkEdgMat, double *d_vlcMat, int *d_elmVtxMat,
                         int lmkNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	strainKernel <<<blkNum, BLKDIM>>> (d_epvMat, d_lmkEdgMat, d_vlcMat, d_elmVtxMat, lmkNum, elmNum);

	return;
}
