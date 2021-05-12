#include "matvec.h"
#include "constants.h"

__global__ void strainKernel(double *d_epvMat, double *d_lmkEdgMat, double *d_vlcMat, int *d_elmVtxMat, 
                             int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec, q30Vec;
		getEdge(q10Vec, q20Vec, q30Vec, d_lmkEdgMat, elmIdx, elmNum);

		int q0Idx = d_elmVtxMat[             elmIdx];
		int q1Idx = d_elmVtxMat[    elmNum + elmIdx];
		int q2Idx = d_elmVtxMat[2 * elmNum + elmIdx];
		int q3Idx = d_elmVtxMat[3 * elmNum + elmIdx];

		vector v0Vec, v1Vec, v2Vec, v3Vec;
		getVector(v0Vec, d_vlcMat, q0Idx, lmkNum);
		getVector(v1Vec, d_vlcMat, q1Idx, lmkNum);
		getVector(v2Vec, d_vlcMat, q2Idx, lmkNum);
		getVector(v3Vec, d_vlcMat, q3Idx, lmkNum);
	
		vector v10Vec, v20Vec, v30Vec;
		vectorSubtract(v10Vec, v1Vec, v0Vec);
		vectorSubtract(v20Vec, v2Vec, v0Vec);
		vectorSubtract(v30Vec, v3Vec, v0Vec);

		// Q = [q1 - q0, q2 - q0, q3 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec, q30Vec);

		// V = [v1 - v0, v2 - v0, v3 - v0]
		matrix VMat;
		columns2Mat(VMat, v10Vec, v20Vec, v30Vec);

		matrix DvMat;
		matMatMul(DvMat, VMat, QInvMat);

		d_epvMat[             elmIdx] = DvMat.x.x;
		d_epvMat[    elmNum + elmIdx] = 0.5 * (DvMat.x.y + DvMat.y.x);
		d_epvMat[2 * elmNum + elmIdx] = 0.5 * (DvMat.x.z + DvMat.z.x);
		d_epvMat[3 * elmNum + elmIdx] = 0.5 * (DvMat.y.x + DvMat.x.y);
		d_epvMat[4 * elmNum + elmIdx] = DvMat.y.y;
		d_epvMat[5 * elmNum + elmIdx] = 0.5 * (DvMat.y.z + DvMat.z.y);
		d_epvMat[6 * elmNum + elmIdx] = 0.5 * (DvMat.z.x + DvMat.x.z);
		d_epvMat[7 * elmNum + elmIdx] = 0.5 * (DvMat.z.y + DvMat.y.z);
		d_epvMat[8 * elmNum + elmIdx] = DvMat.z.z;
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
